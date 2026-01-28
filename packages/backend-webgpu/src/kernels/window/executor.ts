/**
 * Window Kernel Executor
 * 
 * Conv/Pool 操作的主执行器
 * 
 * 职责：
 * 1. 根据 ConvDispatchResult.algorithm 路由到对应算法
 * 2. 处理 Conv 和 Pool 的差异
 * 3. 管理中间缓冲区
 * 
 * @module kernels/window/executor
 */

import type { ITensorHandle } from '@kandle/types';
import type { ConvDispatchResult, ConvVariant, PoolVariant } from './types';
import { Logger } from '@kandle/utils';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { executeIm2ColConv, executeDirectConv, executeWinogradConv } from './algorithms';
import {
    buildPool2dShader,
    computePool2dCacheKey,
    POOL_WORKGROUP_SIZE_X,
    POOL_WORKGROUP_SIZE_Y,
    type Pool2dShaderConfig,
} from './shaderBuilder';
import { createUniformBuffer } from '../../base/uniformUtils';

const logger = new Logger('Window-Executor');

// ============================================================================
// Main Dispatch
// ============================================================================

/**
 * 执行 Conv/Pool 操作
 * 
 * 这是 WindowHandler 调用的主入口点
 * 
 * Note: config is typed as Record<string, unknown> for compatibility with KernelImpl,
 * but it is expected to be a ConvDispatchResult.
 * 
 * 返回值:
 * - 单输出: ITensorHandle (大多数情况)
 * - 多输出: ITensorHandle[] (当 returnIndices=true 时返回 [output, indices])
 */
export function windowExecutor(
    config: Record<string, unknown>,
    ...inputs: ITensorHandle[]
): ITensorHandle | ITensorHandle[] {
    // Cast to ConvDispatchResult for internal use
    const dispatchResult = config as unknown as ConvDispatchResult;
    const variant = dispatchResult.variant;

    // 检测是卷积还是池化
    if (isConvVariant(variant)) {
        return executeConv(dispatchResult, inputs);
    } else {
        return executePool(dispatchResult, inputs);
    }
}


/**
 * 检测是否是卷积操作
 */
function isConvVariant(variant: ConvVariant | PoolVariant): variant is ConvVariant {
    return (
        variant === 'conv1d' ||
        variant === 'conv2d' ||
        variant === 'conv3d' ||
        variant === 'conv_transpose2d' ||
        variant === 'conv_transpose3d'
    );
}

// ============================================================================
// Conv Execution
// ============================================================================

function executeConv(config: ConvDispatchResult, inputs: ITensorHandle[]): ITensorHandle {
    const [input, weight] = inputs;
    const bias = inputs.length > 2 ? inputs[2] : undefined;

    const algorithm = config.algorithm ?? 'im2col';

    logger.debug(`Executing ${config.variant} with algorithm: ${algorithm}`);

    switch (algorithm) {
        case 'direct':
            executeDirectConv(config, input, weight, bias);
            break;

        case 'im2col':
            executeIm2ColConv(config, input, weight, bias);
            break;

        case 'winograd':
            // Winograd F(4,3) 优化的 3x3 卷积
            executeWinogradConv(config, input, weight, bias);
            break;

        case 'fft':
            // FFT 尚未实现，回退到 im2col
            logger.warn('FFT convolution not implemented, falling back to Im2Col');
            executeIm2ColConv(config, input, weight, bias);
            break;

        default:
            throw new Error(`Unknown convolution algorithm: ${algorithm}`);
    }

    return config.output;
}

// ============================================================================
// Pooling Execution
// ============================================================================

function executePool(config: ConvDispatchResult, inputs: ITensorHandle[]): ITensorHandle | ITensorHandle[] {
    const [input] = inputs;
    const variant = config.variant as PoolVariant;

    logger.debug(`Executing ${variant} pool`);

    // 确定池化类型
    const isMaxPool = variant.includes('max');
    const isAdaptive = variant.includes('adaptive');

    if (isAdaptive) {
        executeAdaptivePool(config, input, isMaxPool);
    } else {
        executeStandardPool(config, input, isMaxPool);
    }

    // 当 returnIndices=true 时返回 [output, indices]
    if (config.returnIndices && config.indicesOutput) {
        return [config.output, config.indicesOutput];
    }

    return config.output;
}


// ============================================================================
// Standard Pooling
// ============================================================================

function executeStandardPool(
    config: ConvDispatchResult,
    input: ITensorHandle,
    isMaxPool: boolean
): void {
    const device = WebGPUDeviceManager.device;
    const ndim = config.inputSpatial.length;

    if (ndim === 2) {
        executePool2d(device, config, input, isMaxPool);
    } else if (ndim === 1) {
        executePool1d(device, config, input, isMaxPool);
    } else if (ndim === 3) {
        executePool3d(device, config, input, isMaxPool);
    } else {
        throw new Error(`Unsupported pooling dimensionality: ${ndim}`);
    }
}

// ============================================================================
// Pool2d Implementation
// ============================================================================

function executePool2d(
    device: GPUDevice,
    config: ConvDispatchResult,
    input: ITensorHandle,
    isMaxPool: boolean
): void {
    const { batchSize, inChannels, returnIndices, indicesOutput } = config;
    const [H, W] = config.inputSpatial;
    const [outH, outW] = config.outputSpatial;
    const [kH, kW] = config.kernelSize;
    const [strideH, strideW] = config.stride;
    const [padH, padW] = config.padding;
    const [dilationH, dilationW] = config.dilation;
    const isChannelsLast = config.isChannelsLast;
    const dtype = config.computeDtype;

    // 工业级: 获取输入 strides 和 offset
    const inputStrides = input.strides;
    const inputOffset = input.offset;

    logger.debug(`Pool2d: input strides=[${inputStrides.join(',')}], offset=${inputOffset}`);

    // Build shader config (工业级: 包含 strided 信息)
    const shaderConfig: Pool2dShaderConfig = {
        batchSize,
        channels: inChannels,
        H, W, outH, outW,
        kH, kW, strideH, strideW, padH, padW, dilationH, dilationW,
        isMaxPool, isChannelsLast, dtype,
        returnIndices: returnIndices ?? false,
        // 工业级 Strided 支持
        inputStrides,
        inputOffset,
    };

    // Generate shader
    const shaderCode = buildPool2dShader(shaderConfig);

    // Create uniform buffer
    const uniformData = new Int32Array([
        batchSize, inChannels, H, W,
        outH, outW, kH, kW,
        strideH, strideW, padH, padW,
        dilationH, dilationW, 0, 0  // padding
    ]);

    const uniformBuffer = createUniformBuffer(uniformData.buffer);

    // Get or create pipeline
    const cacheKey = computePool2dCacheKey(shaderConfig);
    let pipeline = WebGPUPipelineManager.getPipeline(cacheKey);
    if (!pipeline) {
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
        WebGPUPipelineManager.registerPipeline(cacheKey, pipeline);
    }

    // Create bind group entries
    const inputBuffer = input.buffer as GPUBuffer;
    const outputBuffer = config.output.buffer as GPUBuffer;

    const bindGroupEntries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: inputBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
    ];

    // 添加 indices 输出缓冲区 (当 returnIndices=true)
    if (returnIndices && indicesOutput) {
        const indicesBuffer = indicesOutput.buffer as GPUBuffer;
        bindGroupEntries.push({
            binding: 3,
            resource: { buffer: indicesBuffer },
        });
    }

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: bindGroupEntries,
    });

    // Dispatch
    const workgroupsX = Math.ceil(outW / POOL_WORKGROUP_SIZE_X);
    const workgroupsY = Math.ceil(outH / POOL_WORKGROUP_SIZE_Y);
    const workgroupsZ = batchSize * inChannels;

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
}


// ============================================================================
// Pool1d Implementation (Simplified)
// ============================================================================

function executePool1d(
    device: GPUDevice,
    config: ConvDispatchResult,
    input: ITensorHandle,
    isMaxPool: boolean
): void {
    // Pool1d 可以视为 Pool2d，其中 H=1
    // 创建一个修改后的 config
    const config2d: ConvDispatchResult = {
        ...config,
        inputSpatial: [1, config.inputSpatial[0]],
        outputSpatial: [1, config.outputSpatial[0]],
        kernelSize: [1, config.kernelSize[0]],
        stride: [1, config.stride[0]],
        padding: [0, config.padding[0]],
        dilation: [1, config.dilation[0]],
    };

    executePool2d(device, config2d, input, isMaxPool);
}

// ============================================================================
// Pool3d Implementation (Simplified)
// ============================================================================

function executePool3d(
    _device: GPUDevice,
    _config: ConvDispatchResult,
    _input: ITensorHandle,
    _isMaxPool: boolean
): void {
    // Pool3d 需要专门的实现
    throw new Error('Pool3d not yet implemented');
}

// ============================================================================
// Adaptive Pooling
// ============================================================================

function executeAdaptivePool(
    config: ConvDispatchResult,
    input: ITensorHandle,
    isMaxPool: boolean
): void {
    // Adaptive pooling 的 kernel/stride 已经在 dispatcher 中计算
    // 直接调用标准池化
    executeStandardPool(config, input, isMaxPool);
}

