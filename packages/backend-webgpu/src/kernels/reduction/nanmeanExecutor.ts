/**
 * NanMean Kernel Executor
 * 
 * Executes nanmean reduction
 * Supports:
 * - Global reduction
 * - Dimensional reduction
 * - Single-pass and Multi-pass
 */

import type { ITensorIterator, DType } from '@kandle/types';
import { Logger } from '@kandle/utils';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { getComputeType } from '../../base/dtype';
import { getGlobalDTypeResolver } from '../../base/DTypeResolver';
import {
    buildDimNanMeanShader,
    buildNaiveGlobalNanMeanShader,
    buildStridedNaiveGlobalNanMeanShader,
    buildGlobalNanMeanStage1Shader,
    buildGlobalNanMeanStage2Shader,
} from './nanmeanShader';
import { createUniformBuffer } from '../../base/uniformUtils';

const logger = new Logger('NanMean-Executor');

const NAIVE_THRESHOLD = 65536;

function getElementByteSize(dtype: DType): number {
    const resolver = getGlobalDTypeResolver();
    const desc = resolver.getDescriptor(dtype);
    return desc.gpuBytesPerElement;
}

// NanMeanStruct size is typically 8 bytes (f32 + u32)
// Even for f16, due to u32 alignment, it's 8 bytes.
const NANMEAN_STRUCT_SIZE = 8;

export function executeNanMean(iter: ITensorIterator, dispatchKey: string): void {
    logger.debug(`Executing NanMean: ${dispatchKey}, isReduction=${iter.isReduction}, reductionAxes=[${iter.reductionAxes.join(',')}]`);

    const isGlobalReduction = iter.outputNumel === 1 &&
        iter.reductionAxes.length === iter.inputShape.length;

    if (isGlobalReduction) {
        executeGlobalNanMean(iter, dispatchKey);
    } else {
        executeDimensionalNanMean(iter, dispatchKey);
    }
}

function executeDimensionalNanMean(iter: ITensorIterator, dispatchKey: string): void {
    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;
    const input = iter.input(0);
    const output = iter.output(0);

    // Use reduction shape length as part of key
    const reductionRank = iter.reductionShape.length;

    const pipelineKey = `nanmean_dim-${iter.computeDtype}-or${iter.outputShape.length}-rr${reductionRank}-wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const shaderCode = buildDimNanMeanShader(iter, workgroupSize);
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    const uniformBuffer = createDimNanMeanUniformBuffer(device, iter);

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
    passEncoder.dispatchWorkgroups(iter.outputNumel);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
}

function executeGlobalNanMean(iter: ITensorIterator, dispatchKey: string): void {
    const numel = iter.reductionNumel;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;

    if (numel <= workgroupSize || numel <= NAIVE_THRESHOLD) {
        executeNaiveGlobalNanMean(iter, dispatchKey);
    } else {
        executeTreeGlobalNanMean(iter, dispatchKey);
    }
}

function executeNaiveGlobalNanMean(iter: ITensorIterator, dispatchKey: string): void {
    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;
    const input = iter.input(0);
    const output = iter.output(0);

    const isContiguous = isInputContiguous(iter);
    const rank = iter.inputShape.length;

    let pipelineKey: string;
    let shaderCode: string;

    if (isContiguous) {
        pipelineKey = `nanmean_naive-${iter.computeDtype}-wg${workgroupSize}`;
        shaderCode = buildNaiveGlobalNanMeanShader(iter, workgroupSize);
    } else {
        pipelineKey = `nanmean_naive_strided-${iter.computeDtype}-r${rank}-wg${workgroupSize}`;
        shaderCode = buildStridedNaiveGlobalNanMeanShader(iter, workgroupSize, rank);
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
    passEncoder.dispatchWorkgroups(1);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
}

function executeTreeGlobalNanMean(iter: ITensorIterator, dispatchKey: string): void {
    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;
    const input = iter.input(0);
    const output = iter.output(0);

    const numel = iter.reductionNumel;
    const outputDtype = output.dtype;
    const computeType = getComputeType(input.dtype);
    const outputType = getComputeType(outputDtype);

    const numWorkgroups = Math.min(
        Math.ceil(numel / workgroupSize),
        256
    );

    // Stage 1
    const stage1Key = `nanmean_s1-${iter.computeDtype}-wg${workgroupSize}`;
    let stage1Pipeline = WebGPUPipelineManager.getPipeline(stage1Key);

    if (!stage1Pipeline) {
        const stage1ShaderCode = buildGlobalNanMeanStage1Shader(iter, workgroupSize);
        const stage1Module = device.createShaderModule({ code: stage1ShaderCode });
        stage1Pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: stage1Module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(stage1Key, stage1Pipeline);
    }

    const partialBufferSize = numWorkgroups * NANMEAN_STRUCT_SIZE;
    const partialBuffer = device.createBuffer({
        size: Math.max(partialBufferSize, NANMEAN_STRUCT_SIZE),
        usage: GPUBufferUsage.STORAGE,
    });

    const stage1Data = new ArrayBuffer(16);
    const stage1View = new DataView(stage1Data);
    stage1View.setUint32(0, numel, true);
    stage1View.setUint32(4, input.offset / getElementByteSize(input.dtype), true);
    const stage1Uniform = createUniformBuffer(stage1Data);

    const stage1BindGroup = device.createBindGroup({
        layout: stage1Pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: input.buffer as GPUBuffer } },
            { binding: 1, resource: { buffer: partialBuffer } },
            { binding: 2, resource: { buffer: stage1Uniform } },
        ],
    });

    // Stage 2
    const stage2Key = `nanmean_s2-${iter.computeDtype}-${outputDtype}-wg${workgroupSize}`;
    let stage2Pipeline = WebGPUPipelineManager.getPipeline(stage2Key);

    if (!stage2Pipeline) {
        const stage2ShaderCode = buildGlobalNanMeanStage2Shader(computeType, outputType, workgroupSize);
        const stage2Module = device.createShaderModule({ code: stage2ShaderCode });
        stage2Pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: stage2Module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(stage2Key, stage2Pipeline);
    }

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
}

// ============================================================================
// Uniform Buffers (Reused patterns)
// ============================================================================

function createDimNanMeanUniformBuffer(
    device: GPUDevice,
    iter: ITensorIterator
): GPUBuffer {
    const input = iter.input(0);
    const output = iter.output(0);
    const inputDtype = input.dtype;
    const outputDtype = output.dtype;

    // 192 bytes like logsumexp/welford
    const bufferSize = 192;
    const data = new ArrayBuffer(bufferSize);
    const u32View = new Uint32Array(data);

    u32View[0] = iter.outputNumel;
    u32View[1] = iter.reductionNumel;
    u32View[2] = 0;
    u32View[3] = iter.inputShape.length;

    const inputShape = [...iter.inputShape];
    const outputShape = [...iter.outputShape];
    const inputStrides = [...input.tensorHandle.strides]; // Use handle strides for parallel dims calc
    const reductionStrides = input.reductionStrides ? [...input.reductionStrides] : [];
    const reductionShape = [...iter.reductionShape];

    for (let i = 0; i < 4; i++) u32View[4 + i] = inputShape[i] ?? 0;
    for (let i = 0; i < 4; i++) u32View[8 + i] = inputShape[4 + i] ?? 0;
    for (let i = 0; i < 4; i++) u32View[12 + i] = outputShape[i] ?? 0;
    for (let i = 0; i < 4; i++) u32View[16 + i] = outputShape[4 + i] ?? 0;
    for (let i = 0; i < 4; i++) u32View[20 + i] = inputStrides[i] ?? 0;
    for (let i = 0; i < 4; i++) u32View[24 + i] = inputStrides[4 + i] ?? 0;
    for (let i = 0; i < 4; i++) u32View[28 + i] = reductionStrides[i] ?? 0;
    for (let i = 0; i < 4; i++) u32View[32 + i] = reductionStrides[4 + i] ?? 0;
    for (let i = 0; i < 4; i++) u32View[36 + i] = reductionShape[i] ?? 0;
    for (let i = 0; i < 4; i++) u32View[40 + i] = reductionShape[4 + i] ?? 0;

    u32View[44] = input.offset / getElementByteSize(inputDtype);
    u32View[45] = output.offset / getElementByteSize(outputDtype);

    return createUniformBuffer(data);
}

function createNaiveGlobalUniformBuffer(
    device: GPUDevice,
    iter: ITensorIterator
): GPUBuffer {
    const input = iter.input(0);
    const output = iter.output(0);

    // 16 bytes
    const data = new ArrayBuffer(16);
    const dataView = new DataView(data);
    dataView.setUint32(0, iter.reductionNumel, true);
    dataView.setUint32(4, input.offset / getElementByteSize(input.dtype), true);
    dataView.setUint32(8, output.offset / getElementByteSize(output.dtype), true);

    return createUniformBuffer(data);
}

function createStridedGlobalUniformBuffer(
    device: GPUDevice,
    iter: ITensorIterator
): GPUBuffer {
    const input = iter.input(0);
    const output = iter.output(0);

    // 80 bytes
    const bufferSize = 80;
    const data = new ArrayBuffer(bufferSize);
    const u32View = new Uint32Array(data);

    const shape = [...iter.inputShape];
    const strides = [...input.tensorHandle.strides];
    const rank = shape.length;

    u32View[0] = iter.reductionNumel;
    u32View[1] = rank;
    u32View[2] = input.offset / getElementByteSize(input.dtype);
    u32View[3] = output.offset / getElementByteSize(output.dtype);

    for (let i = 0; i < 4; i++) u32View[4 + i] = shape[i] ?? 0;
    for (let i = 0; i < 4; i++) u32View[8 + i] = shape[4 + i] ?? 0;
    for (let i = 0; i < 4; i++) u32View[12 + i] = strides[i] ?? 0;
    for (let i = 0; i < 4; i++) u32View[16 + i] = strides[4 + i] ?? 0;

    return createUniformBuffer(data);
}

function isInputContiguous(iter: ITensorIterator): boolean {
    const input = iter.input(0);
    if (input.offset !== 0) return false;

    const shape = iter.inputShape;
    const strides = input.tensorHandle.strides;

    let expectedStride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
        if (strides[i] !== expectedStride) return false;
        expectedStride *= shape[i];
    }
    return true;
}
