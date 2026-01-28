/**
 * LogSumExp Kernel Executor
 * 
 * 执行 logsumexp reduction
 * 支持:
 * - Global reduction (全部元素归约到标量)
 * - Dimensional reduction (沿指定维度归约)
 * - Single-pass (小数据) 和 Multi-pass (大数据)
 * 
 * 参考: PyTorch ATen/native/ReduceOps.cpp
 */

import type { ITensorIterator, DType } from '@kandle/types';
import { Logger } from '@kandle/utils';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { getComputeType } from '../../base/dtype';
import { getGlobalDTypeResolver } from '../../base/DTypeResolver';
import {
    buildDimLogSumExpShader,
    buildNaiveGlobalLogSumExpShader,
    buildStridedNaiveGlobalLogSumExpShader,
    buildGlobalLogSumExpStage1Shader,
    buildGlobalLogSumExpStage2Shader,
} from './logsumexpShader';
import { createUniformBuffer } from '../../base/uniformUtils';

const logger = new Logger('LogSumExp-Executor');

// 阈值常量
const NAIVE_THRESHOLD = 65536; // 小于此值使用单 pass

// ============================================================================
// Helper: Get byte size per element
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
 * 执行 LogSumExp reduction
 */
export function executeLogSumExp(iter: ITensorIterator, dispatchKey: string): void {
    logger.debug(`Executing LogSumExp: ${dispatchKey}, isReduction=${iter.isReduction}, reductionAxes=[${iter.reductionAxes.join(',')}]`);

    // 区分 global 和 dimensional reduction
    const isGlobalReduction = iter.outputNumel === 1 &&
        iter.reductionAxes.length === iter.inputShape.length;

    if (isGlobalReduction) {
        executeGlobalLogSumExp(iter, dispatchKey);
    } else {
        executeDimensionalLogSumExp(iter, dispatchKey);
    }
}

// ============================================================================
// Dimensional Reduction
// ============================================================================

/**
 * 执行 Dimensional LogSumExp Reduction
 * 每个 workgroup 处理一个输出元素
 */
function executeDimensionalLogSumExp(iter: ITensorIterator, dispatchKey: string): void {
    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;
    const input = iter.input(0);
    const output = iter.output(0);

    logger.debug(`Dim LogSumExp: outputNumel=${iter.outputNumel}, reductionNumel=${iter.reductionNumel}`);

    // Pipeline key
    const pipelineKey = `logsumexp_dim-${iter.computeDtype}-or${iter.outputShape.length}-rr${iter.reductionShape.length}-wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        // 生成 shader
        const shaderCode = buildDimLogSumExpShader(iter, workgroupSize);
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
        logger.debug(`Created dim LogSumExp pipeline: ${pipelineKey}`);
    }

    // 创建 uniform buffer
    const uniformBuffer = createDimLogSumExpUniformBuffer(device, iter);

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
    passEncoder.dispatchWorkgroups(iter.outputNumel); // 每个 workgroup 处理一个输出
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);

    logger.debug(`Dim LogSumExp complete: ${iter.outputNumel} workgroups`);
}

// ============================================================================
// Global Reduction
// ============================================================================

/**
 * 执行 Global LogSumExp Reduction
 * 选择 naive (单 pass) 或 tree (多 pass)
 */
function executeGlobalLogSumExp(iter: ITensorIterator, dispatchKey: string): void {
    const numel = iter.reductionNumel;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;

    if (numel <= workgroupSize) {
        executeNaiveGlobalLogSumExp(iter, dispatchKey);
    } else if (numel <= NAIVE_THRESHOLD) {
        executeNaiveGlobalLogSumExp(iter, dispatchKey);
    } else {
        executeTreeGlobalLogSumExp(iter, dispatchKey);
    }
}

/**
 * Naive Global LogSumExp (单 workgroup)
 */
function executeNaiveGlobalLogSumExp(iter: ITensorIterator, dispatchKey: string): void {
    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;
    const input = iter.input(0);
    const output = iter.output(0);

    const numel = iter.reductionNumel;
    const isContiguous = isInputContiguous(iter);
    const rank = iter.inputShape.length;

    logger.debug(`Naive Global LogSumExp: numel=${numel}, contiguous=${isContiguous}`);

    let pipelineKey: string;
    let shaderCode: string;

    if (isContiguous) {
        pipelineKey = `logsumexp_naive-${iter.computeDtype}-wg${workgroupSize}`;
        shaderCode = buildNaiveGlobalLogSumExpShader(iter, workgroupSize);
    } else {
        pipelineKey = `logsumexp_naive_strided-${iter.computeDtype}-r${rank}-wg${workgroupSize}`;
        shaderCode = buildStridedNaiveGlobalLogSumExpShader(iter, workgroupSize, rank);
    }

    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // 创建 uniform buffer
    const uniformBuffer = isContiguous
        ? createNaiveGlobalUniformBuffer(device, iter)
        : createStridedGlobalUniformBuffer(device, iter);

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
    passEncoder.dispatchWorkgroups(1); // 单 workgroup
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
}

/**
 * Tree Global LogSumExp (多 workgroup, 两阶段)
 */
function executeTreeGlobalLogSumExp(iter: ITensorIterator, dispatchKey: string): void {
    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;
    const input = iter.input(0);
    const output = iter.output(0);

    const numel = iter.reductionNumel;
    const inputDtype = input.dtype;
    const outputDtype = output.dtype;
    const computeType = getComputeType(inputDtype);
    const outputType = getComputeType(outputDtype);

    // 计算 workgroup 数量
    const numWorkgroups = Math.min(
        Math.ceil(numel / workgroupSize),
        256 // 最多 256 个 workgroups
    );

    logger.debug(`Tree Global LogSumExp: numel=${numel}, numWorkgroups=${numWorkgroups}`);

    // Stage 1 Pipeline
    const stage1Key = `logsumexp_s1-${iter.computeDtype}-wg${workgroupSize}`;
    let stage1Pipeline = WebGPUPipelineManager.getPipeline(stage1Key);

    if (!stage1Pipeline) {
        const stage1ShaderCode = buildGlobalLogSumExpStage1Shader(iter, workgroupSize);
        const stage1Module = device.createShaderModule({ code: stage1ShaderCode });
        stage1Pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: stage1Module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(stage1Key, stage1Pipeline);
    }

    // 中间 buffer: numWorkgroups * sizeof(vec2<f32>) = numWorkgroups * 8 bytes
    const partialBufferSize = numWorkgroups * 8; // vec2<f32> = 8 bytes
    const partialBuffer = device.createBuffer({
        size: Math.max(partialBufferSize, 8),
        usage: GPUBufferUsage.STORAGE,
    });

    // Stage 1 uniform
    const stage1Data = new ArrayBuffer(16);
    const stage1View = new DataView(stage1Data);
    stage1View.setUint32(0, numel, true);
    stage1View.setUint32(4, input.offset / getElementByteSize(inputDtype), true);
    const stage1Uniform = createUniformBuffer(stage1Data);

    const stage1BindGroup = device.createBindGroup({
        layout: stage1Pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: input.buffer as GPUBuffer } },
            { binding: 1, resource: { buffer: partialBuffer } },
            { binding: 2, resource: { buffer: stage1Uniform } },
        ],
    });

    // Stage 2 Pipeline
    const stage2Key = `logsumexp_s2-${iter.computeDtype}-${outputDtype}-wg${workgroupSize}`;
    let stage2Pipeline = WebGPUPipelineManager.getPipeline(stage2Key);

    if (!stage2Pipeline) {
        const stage2ShaderCode = buildGlobalLogSumExpStage2Shader(computeType, outputType, workgroupSize);
        const stage2Module = device.createShaderModule({ code: stage2ShaderCode });
        stage2Pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: stage2Module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(stage2Key, stage2Pipeline);
    }

    // Stage 2 uniform
    const stage2Data = new ArrayBuffer(16);
    const stage2View = new DataView(stage2Data);
    stage2View.setUint32(0, numWorkgroups, true);
    stage2View.setUint32(4, output.offset / getElementByteSize(outputDtype), true);
    const stage2Uniform = createUniformBuffer(stage2Data);

    const stage2BindGroup = device.createBindGroup({
        layout: stage2Pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: partialBuffer } },
            { binding: 1, resource: { buffer: output.buffer as GPUBuffer } },
            { binding: 2, resource: { buffer: stage2Uniform } },
        ],
    });

    // 执行两阶段
    const commandEncoder = device.createCommandEncoder();

    const pass1 = commandEncoder.beginComputePass();
    pass1.setPipeline(stage1Pipeline);
    pass1.setBindGroup(0, stage1BindGroup);
    pass1.dispatchWorkgroups(numWorkgroups);
    pass1.end();

    const pass2 = commandEncoder.beginComputePass();
    pass2.setPipeline(stage2Pipeline);
    pass2.setBindGroup(0, stage2BindGroup);
    pass2.dispatchWorkgroups(1);
    pass2.end();

    device.queue.submit([commandEncoder.finish()]);

    logger.debug(`Tree Global LogSumExp complete: ${numWorkgroups} -> 1`);
}

// ============================================================================
// Uniform Buffer Creation
// ============================================================================

/**
 * 创建 Dimensional LogSumExp uniform buffer
 */
function createDimLogSumExpUniformBuffer(
    device: GPUDevice,
    iter: ITensorIterator
): GPUBuffer {
    const input = iter.input(0);
    const output = iter.output(0);
    const inputDtype = input.dtype;
    const outputDtype = output.dtype;

    // Layout: 参考 shaderBuilder 中的 Uniforms struct
    // Total: 192 bytes
    const bufferSize = 192;
    const data = new ArrayBuffer(bufferSize);
    const u32View = new Uint32Array(data);

    // Basic params
    u32View[0] = iter.outputNumel;
    u32View[1] = iter.reductionNumel;
    u32View[2] = 0; // padding
    u32View[3] = iter.inputShape.length;

    // Shapes and strides (vec4 aligned)
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

/**
 * 创建 Naive Global uniform buffer (连续输入)
 */
function createNaiveGlobalUniformBuffer(
    device: GPUDevice,
    iter: ITensorIterator
): GPUBuffer {
    const input = iter.input(0);
    const output = iter.output(0);
    const inputDtype = input.dtype;
    const outputDtype = output.dtype;

    // Layout: numel(4), inputOffset(4), outputOffset(4), pad(4) = 16 bytes
    const data = new ArrayBuffer(16);
    const dataView = new DataView(data);
    dataView.setUint32(0, iter.reductionNumel, true);
    dataView.setUint32(4, input.offset / getElementByteSize(inputDtype), true);
    dataView.setUint32(8, output.offset / getElementByteSize(outputDtype), true);

    return createUniformBuffer(data);
}

/**
 * 创建 Strided Global uniform buffer (非连续输入)
 */
function createStridedGlobalUniformBuffer(
    device: GPUDevice,
    iter: ITensorIterator
): GPUBuffer {
    const input = iter.input(0);
    const output = iter.output(0);
    const inputDtype = input.dtype;
    const outputDtype = output.dtype;

    // Layout: numel(4), rank(4), inputOffset(4), outputOffset(4),
    //         shape(16), shape2(16), strides(16), strides2(16)
    // Total: 80 bytes
    const bufferSize = 80;
    const data = new ArrayBuffer(bufferSize);
    const u32View = new Uint32Array(data);

    const shape = [...iter.inputShape];
    const strides = [...input.tensorHandle.strides];
    const rank = shape.length;

    u32View[0] = iter.reductionNumel;
    u32View[1] = rank;
    u32View[2] = input.offset / getElementByteSize(inputDtype);
    u32View[3] = output.offset / getElementByteSize(outputDtype);

    // shape (offset 16)
    for (let i = 0; i < 4; i++) u32View[4 + i] = shape[i] ?? 0;
    // shape2 (offset 32)
    for (let i = 0; i < 4; i++) u32View[8 + i] = shape[4 + i] ?? 0;
    // strides (offset 48)
    for (let i = 0; i < 4; i++) u32View[12 + i] = strides[i] ?? 0;
    // strides2 (offset 64)
    for (let i = 0; i < 4; i++) u32View[16 + i] = strides[4 + i] ?? 0;

    return createUniformBuffer(data);
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * 检查输入是否连续
 */
function isInputContiguous(iter: ITensorIterator): boolean {
    const input = iter.input(0);
    if (input.offset !== 0) return false;

    const shape = iter.inputShape;
    const strides = input.tensorHandle.strides;

    // 检查是否为标准 row-major strides
    let expectedStride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
        if (strides[i] !== expectedStride) return false;
        expectedStride *= shape[i];
    }
    return true;
}
