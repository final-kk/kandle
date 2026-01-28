/**
 * Norm Kernel Executor
 * 
 * 执行 Lp 范数 reduction
 * 支持:
 * - Global reduction (全部元素归约到标量)
 * - Dimensional reduction (沿指定维度归约)
 * - 特殊 p 值优化 (p=0,1,2,inf,-inf)
 * 
 * 参考: PyTorch torch.linalg.vector_norm
 */

import type { ITensorIterator, DType } from '@kandle/types';
import { Logger } from '@kandle/utils';
import { WebGPUDeviceManager } from '../../../base/device';
import { WebGPUPipelineManager } from '../../../pipelines/WebGPUPipelineManager';
import { getGlobalDTypeResolver } from '../../../base/DTypeResolver';
import {
    buildDimNormShader,
    buildNaiveGlobalL2NormShader,
    buildNaiveGlobalNormShader,
} from './shaderBuilder';
import { NORM_OPS } from './ops';
import { getNormType, type NormOrd } from './types';
import { createUniformBuffer } from '../../../base/uniformUtils';

const logger = new Logger('Norm-Executor');

// ============================================================================
// Helper
// ============================================================================

function getElementByteSize(dtype: DType): number {
    const resolver = getGlobalDTypeResolver();
    const desc = resolver.getDescriptor(dtype);
    return desc.gpuBytesPerElement;
}

// ============================================================================
// Main Entry Point
// ============================================================================

/**
 * 执行 Norm reduction
 */
export function executeNorm(iter: ITensorIterator, dispatchKey: string): void {
    const config = NORM_OPS[dispatchKey];
    if (!config) {
        throw new Error(`Unknown Norm operation: ${dispatchKey}`);
    }

    // 获取 p 参数
    const p: NormOrd = (iter as any).getScalarArg?.('p') ?? 2;
    const normType = getNormType(p);

    logger.debug(`Executing Norm: ${dispatchKey}, p=${p}, normType=${normType}, reductionAxes=[${iter.reductionAxes.join(',')}]`);

    // 区分 global 和 dimensional reduction
    const isGlobalReduction = iter.outputNumel === 1 &&
        iter.reductionAxes.length === iter.inputShape.length;

    if (isGlobalReduction) {
        executeGlobalNorm(iter, normType, p);
    } else {
        executeDimensionalNorm(iter, normType, p);
    }
}

// ============================================================================
// Dimensional Reduction
// ============================================================================

function executeDimensionalNorm(iter: ITensorIterator, normType: ReturnType<typeof getNormType>, p: NormOrd): void {
    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;
    const input = iter.input(0);
    const output = iter.output(0);

    logger.debug(`Dim Norm: outputNumel=${iter.outputNumel}, reductionNumel=${iter.reductionNumel}, normType=${normType}`);

    // Pipeline key
    const pipelineKey = `norm_dim.${normType}-${iter.computeDtype}-or${iter.outputShape.length}-rr${iter.reductionShape.length}-wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const shaderCode = buildDimNormShader(iter, workgroupSize, normType);
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
        logger.debug(`Created dim Norm pipeline: ${pipelineKey}`);
    }

    // 创建 uniform buffer
    const uniformBuffer = createDimNormUniformBuffer(device, iter, normType === 'general' ? (p as number) : 2);

    // 创建 bind group
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: input.buffer as GPUBuffer } },
            { binding: 1, resource: { buffer: output.buffer as GPUBuffer } },
            { binding: 2, resource: { buffer: uniformBuffer } },
        ],
    });

    // 执行
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(iter.outputNumel);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);

    logger.debug(`Dim Norm complete: ${iter.outputNumel} workgroups`);
}

// ============================================================================
// Global Reduction
// ============================================================================

function executeGlobalNorm(iter: ITensorIterator, normType: ReturnType<typeof getNormType>, p: NormOrd): void {
    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;
    const input = iter.input(0);
    const output = iter.output(0);
    const numel = iter.reductionNumel;

    logger.debug(`Global Norm: numel=${numel}, normType=${normType}, p=${p}`);

    // 对于 global reduction，使用 dimensional shader （视所有元素为一个 workgroup）
    // 这样可以复用所有 normType 的 shader 逻辑
    const pipelineKey = `norm_naive.${normType}-${iter.computeDtype}-wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        // 使用 dimensional shader (1 outputNumel, numel reductionNumel)
        const shaderCode = buildNaiveGlobalNormShader(iter, workgroupSize, normType, p);
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // Uniform buffer: numel(4), inputOffset(4), outputOffset(4), p(4) = 16 bytes
    const data = new ArrayBuffer(16);
    const u32View = new Uint32Array(data);
    const f32View = new Float32Array(data);
    u32View[0] = numel;
    u32View[1] = input.offset / getElementByteSize(input.dtype);
    u32View[2] = output.offset / getElementByteSize(output.dtype);
    f32View[3] = typeof p === 'number' && Number.isFinite(p) ? p : 2; // p 值，默认 2
    const uniformBuffer = createUniformBuffer(data);

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: input.buffer as GPUBuffer } },
            { binding: 1, resource: { buffer: output.buffer as GPUBuffer } },
            { binding: 2, resource: { buffer: uniformBuffer } },
        ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(1);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
}

// ============================================================================
// Uniform Buffer Creation
// ============================================================================

function createDimNormUniformBuffer(
    device: GPUDevice,
    iter: ITensorIterator,
    p: number
): GPUBuffer {
    const input = iter.input(0);
    const output = iter.output(0);
    const inputDtype = input.dtype;
    const outputDtype = output.dtype;

    // Layout: 192 bytes (与 logsumexp/welford 相同)
    const bufferSize = 192;
    const data = new ArrayBuffer(bufferSize);
    const u32View = new Uint32Array(data);
    const f32View = new Float32Array(data);

    // Basic params
    u32View[0] = iter.outputNumel;
    u32View[1] = iter.reductionNumel;
    f32View[2] = p;  // p 值 (for general norm)
    u32View[3] = iter.inputShape.length;

    // Shapes and strides
    const inputShape = [...iter.inputShape];
    const outputShape = [...iter.outputShape];
    const inputStrides = [...input.strides];
    const reductionStrides = input.reductionStrides ? [...input.reductionStrides] : [];
    const reductionShape = [...iter.reductionShape];

    // inputShape (offset 16)
    for (let i = 0; i < 4; i++) u32View[4 + i] = inputShape[i] ?? 0;
    // inputShape2 (offset 32)
    for (let i = 0; i < 4; i++) u32View[8 + i] = inputShape[4 + i] ?? 0;
    // outputShape (offset 48)
    for (let i = 0; i < 4; i++) u32View[12 + i] = outputShape[i] ?? 0;
    // outputShape2 (offset 64)
    for (let i = 0; i < 4; i++) u32View[16 + i] = outputShape[4 + i] ?? 0;
    // inputStrides (offset 80)
    for (let i = 0; i < 4; i++) u32View[20 + i] = inputStrides[i] ?? 0;
    // inputStrides2 (offset 96)
    for (let i = 0; i < 4; i++) u32View[24 + i] = inputStrides[4 + i] ?? 0;
    // reductionStrides (offset 112)
    for (let i = 0; i < 4; i++) u32View[28 + i] = reductionStrides[i] ?? 0;
    // reductionStrides2 (offset 128)
    for (let i = 0; i < 4; i++) u32View[32 + i] = reductionStrides[4 + i] ?? 0;
    // reductionShape (offset 144)
    for (let i = 0; i < 4; i++) u32View[36 + i] = reductionShape[i] ?? 0;
    // reductionShape2 (offset 160)
    for (let i = 0; i < 4; i++) u32View[40 + i] = reductionShape[4 + i] ?? 0;
    // inputOffset, outputOffset (offset 176)
    u32View[44] = input.offset / getElementByteSize(inputDtype);
    u32View[45] = output.offset / getElementByteSize(outputDtype);

    return createUniformBuffer(data);
}
