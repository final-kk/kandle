/**
 * FlashAttention Executor
 * 
 * 管理 FlashAttention kernel 的执行:
 * - Pipeline 缓存
 * - Uniform buffer 创建
 * - BindGroup 创建
 * - Dispatch 计算
 */

import type { ITensorHandle } from '@kandle/types';
import { Logger } from '@kandle/utils';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { getGlobalDTypeResolver } from '../../base/DTypeResolver';
import { UniformBufferPool } from '../../base/UniformBufferPool';
import {
    FlashAttentionConfig,
    selectTileConfig,
    createFlashAttentionUniformBuffer,
    computeFlashAttentionCacheKey,
    FLASH_ATTENTION_UNIFORM_LAYOUT,
} from './types';
import { buildFlashAttentionShader } from './builder';

const logger = new Logger('FlashAttention-Executor');

// ============================================================================
// Main Executor
// ============================================================================

/**
 * 执行 FlashAttention kernel
 * 
 * @param query Query tensor [batch, numHeadsQ, seqLenQ, headDim]
 * @param key Key tensor [batch, numHeadsKV, seqLenKV, headDim]
 * @param value Value tensor [batch, numHeadsKV, seqLenKV, headDim]
 * @param output Output tensor [batch, numHeadsQ, seqLenQ, headDim]
 * @param config FlashAttention 配置
 */
export function executeFlashAttention(
    query: ITensorHandle,
    key: ITensorHandle,
    value: ITensorHandle,
    output: ITensorHandle,
    config: FlashAttentionConfig
): void {
    const device = WebGPUDeviceManager.device;
    const { batchSize, numHeadsQ, numHeadsKV, seqLenQ, seqLenKV, headDim, scale, isCausal, dtype } = config;

    logger.debug(`FlashAttention: batch=${batchSize}, headsQ=${numHeadsQ}, headsKV=${numHeadsKV}, seqQ=${seqLenQ}, seqKV=${seqLenKV}, headDim=${headDim}, causal=${isCausal}`);

    // 1. 选择 tile 配置
    const tileConfig = selectTileConfig(headDim, seqLenQ, seqLenKV);
    logger.debug(`Tile config: blockQ=${tileConfig.blockSizeQ}, blockKV=${tileConfig.blockSizeKV}, wgSize=[${tileConfig.workgroupSizeX}, ${tileConfig.workgroupSizeY}]`);

    // 2. 计算 GQA ratio
    const gqaRatio = numHeadsQ / numHeadsKV;
    if (!Number.isInteger(gqaRatio)) {
        throw new Error(`FlashAttention: numHeadsQ (${numHeadsQ}) must be divisible by numHeadsKV (${numHeadsKV})`);
    }

    // 3. 获取或创建 pipeline
    const pipelineKey = computeFlashAttentionCacheKey(
        dtype, headDim, tileConfig.blockSizeQ, tileConfig.blockSizeKV, isCausal, numHeadsQ, numHeadsKV
    );
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    // 4. BindGroupLayout
    const bindGroupLayoutKey = 'attention.flash-layout';
    let bindGroupLayout = WebGPUPipelineManager.getBindGroupLayout(bindGroupLayoutKey);

    if (!bindGroupLayout) {
        bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },      // uniforms
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // Q
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // K
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // V
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },     // O
            ],
        });
        WebGPUPipelineManager.registerBindGroupLayout(bindGroupLayoutKey, bindGroupLayout);
    }

    // 5. 创建 pipeline (如果不存在)
    if (!pipeline) {
        const shaderConfig = {
            dtype,
            headDim,
            blockSizeQ: tileConfig.blockSizeQ,
            blockSizeKV: tileConfig.blockSizeKV,
            isCausal,
            gqaRatio,
        };

        const shaderCode = buildFlashAttentionShader(shaderConfig, tileConfig);
        logger.debug('Generated FlashAttention shader');

        const module = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // 6. 创建 Uniform Buffer
    const resolver = getGlobalDTypeResolver();
    const bytesPerElement = resolver.getDescriptor(dtype).gpuBytesPerElement;

    const offsetQ = query.offset / bytesPerElement;
    const offsetK = key.offset / bytesPerElement;
    const offsetV = value.offset / bytesPerElement;
    const offsetO = output.offset / bytesPerElement;

    const uniformData = createFlashAttentionUniformBuffer({
        batchSize,
        numHeadsQ,
        numHeadsKV,
        headDim,
        seqLenQ,
        seqLenKV,
        blockSizeQ: tileConfig.blockSizeQ,
        blockSizeKV: tileConfig.blockSizeKV,
        scale,
        offsetQ,
        offsetK,
        offsetV,
        offsetO,
        isCausal,
    });

    const uniformBuffer = UniformBufferPool.getInstance().acquire(
        FLASH_ATTENTION_UNIFORM_LAYOUT.size,
        uniformData,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST  // Uniform buffer needs UNIFORM usage
    );

    // 7. 创建 BindGroup
    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: query.buffer as GPUBuffer } },
            { binding: 2, resource: { buffer: key.buffer as GPUBuffer } },
            { binding: 3, resource: { buffer: value.buffer as GPUBuffer } },
            { binding: 4, resource: { buffer: output.buffer as GPUBuffer } },
        ],
    });

    // 8. 计算 dispatch 维度
    // 每个 workgroup 处理一个 Q block
    const numQBlocks = Math.ceil(seqLenQ / tileConfig.blockSizeQ);
    const dispatchX = numQBlocks;
    const dispatchY = numHeadsQ;
    const dispatchZ = batchSize;

    logger.debug(`Dispatch: [${dispatchX}, ${dispatchY}, ${dispatchZ}]`);

    // 9. 执行
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
}

// ============================================================================
// Higher-level API
// ============================================================================

/**
 * 从 tensor 形状推断 FlashAttention 配置并执行
 * 
 * 支持的输入形状:
 * - 4D: [batch, num_heads, seq_len, head_dim]
 * - 3D: [num_heads, seq_len, head_dim] (batch=1)
 * - 2D: [seq_len, head_dim] (batch=1, heads=1)
 */
export function flashAttention(
    query: ITensorHandle,
    key: ITensorHandle,
    value: ITensorHandle,
    output: ITensorHandle,
    scale: number,
    isCausal: boolean,
): void {
    const qShape = query.shape;
    const kShape = key.shape;

    // 解析维度
    let batchSize: number, numHeadsQ: number, seqLenQ: number, headDim: number;
    let numHeadsKV: number, seqLenKV: number;

    if (qShape.length === 4) {
        [batchSize, numHeadsQ, seqLenQ, headDim] = qShape;
    } else if (qShape.length === 3) {
        batchSize = 1;
        [numHeadsQ, seqLenQ, headDim] = qShape;
    } else if (qShape.length === 2) {
        batchSize = 1;
        numHeadsQ = 1;
        [seqLenQ, headDim] = qShape;
    } else {
        throw new Error(`FlashAttention: unsupported query shape ${qShape}`);
    }

    if (kShape.length === 4) {
        [, numHeadsKV, seqLenKV,] = kShape;
    } else if (kShape.length === 3) {
        [numHeadsKV, seqLenKV,] = kShape;
    } else if (kShape.length === 2) {
        numHeadsKV = 1;
        [seqLenKV,] = kShape;
    } else {
        throw new Error(`FlashAttention: unsupported key shape ${kShape}`);
    }

    const config: FlashAttentionConfig = {
        batchSize,
        numHeadsQ,
        numHeadsKV,
        seqLenQ,
        seqLenKV,
        headDim,
        scale,
        isCausal,
        dtype: query.dtype,
    };

    executeFlashAttention(query, key, value, output, config);
}
