/**
 * Reduction Kernel Executor (v5)
 * 
 * Unified reduction executor using REDUCTION_OPS registry
 * Supports:
 * - Global reduction (all elements to scalar)
 * - Dimensional reduction (reduce along specific axes)
 * 
 * Uses correct WGSL memory alignment with vec4 types
 */

import { ITensorIterator } from '@kandle/types';
import { Logger } from '@kandle/utils';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { getGlobalDTypeResolver } from '../../base/DTypeResolver';
import { getComputeType } from '../../base/dtype';
import { REDUCTION_OPS } from './ops';
import {
    buildDimReductionShader,
    buildGlobalReductionStage1,
    buildStridedGlobalReductionStage1,
    buildGlobalReductionStage2,
    buildNaiveReductionShader,
    buildStridedNaiveReductionShader
} from './shaderBuilder';

import { executeLogSumExp } from './logsumexpExecutor';
import { executeNanMean } from './nanmeanExecutor';
import { createUniformBuffer } from '../../base/uniformUtils';

const logger = new Logger('Reduction-Executor');

/**
 * Main reduction executor entry point
 */
export function executeReduction(iter: ITensorIterator, dispatchKey: string): void {
    if (dispatchKey === 'logsumexp') {
        executeLogSumExp(iter, dispatchKey);
        return;
    }
    if (dispatchKey === 'nanmean') {
        executeNanMean(iter, dispatchKey);
        return;
    }

    const opConfig = REDUCTION_OPS[dispatchKey];
    if (!opConfig) {
        throw new Error(`Unknown reduction operation: ${dispatchKey}`);
    }

    if (!iter.isReduction) {
        throw new Error(`Iterator must be a reduction operation`);
    }

    // Check if this is global reduction or dimensional reduction
    const isGlobalReduction = iter.outputShape.length === 0 ||
        (iter.outputShape.length === 1 && iter.outputShape[0] === 1 && iter.reductionShape.length > 0);

    if (isGlobalReduction) {
        executeGlobalReduction(iter, dispatchKey);
    } else {
        executeDimensionalReduction(iter, dispatchKey);
    }
}

/**
 * Execute dimensional reduction (sum(dim=0), mean(dim=1), etc.)
 * Each workgroup handles one output element
 */
function executeDimensionalReduction(iter: ITensorIterator, dispatchKey: string): void {
    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;

    const outputNumel = iter.outputNumel;
    const reductionNumel = iter.reductionNumel;

    logger.debug(`Dim reduction: ${outputNumel} output elements, ${reductionNumel} reduction elements each`);

    // Generate shader
    const shaderCode = buildDimReductionShader(iter, dispatchKey, workgroupSize);

    // Pipeline key
    const pipelineKey = `reduction_dim.${dispatchKey}-${iter.computeDtype}-or${iter.outputShape.length}-rr${iter.reductionShape.length}-wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const module = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
        logger.debug(`Created dim reduction pipeline: ${pipelineKey}`);
    }

    // Create uniform buffer
    const uniformBuffer = createDimReductionUniformBuffer(device, iter);

    // Get buffers
    const inputBuffer = iter.input(0).buffer as GPUBuffer;
    const outputBuffer = iter.output().buffer as GPUBuffer;

    // Create bind group
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: inputBuffer } },
            { binding: 2, resource: { buffer: outputBuffer } },
        ],
    });

    // Dispatch: one workgroup per output element
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(outputNumel);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    logger.debug(`Dim reduction complete: ${outputNumel} workgroups`);
}

/**
 * Execute global reduction (all elements to scalar)
 */
function executeGlobalReduction(iter: ITensorIterator, dispatchKey: string): void {
    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;
    const numel = iter.reductionNumel;
    const opConfig = REDUCTION_OPS[dispatchKey];

    // Strategy selection
    if (numel <= workgroupSize) {
        // Small data: use naive single-workgroup reduction
        executeNaiveGlobalReduction(iter, dispatchKey);
    } else {
        // Large data: use tree reduction
        executeTreeGlobalReduction(iter, dispatchKey);
    }
}

/**
 * Check if input tensor is contiguous for global reduction purposes
 * A tensor is contiguous if strides follow row-major order and offset is 0
 */
function isInputContiguousForGlobalReduction(iter: ITensorIterator): boolean {
    const input = iter.input(0);

    // Check offset
    if (input.offset !== 0) {
        return false;
    }

    // For global reduction, we need to check the original input shape/strides
    // The input.strides here are reductionStrides (since it's a reduction op)
    // We should check if the original tensor is contiguous

    // Get the original shape from inputShape (before reduction)
    const shape = iter.inputShape;
    const strides = input.tensorHandle.strides;

    // Check row-major contiguity
    let expectedStride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
        if (shape[i] > 1 && strides[i] !== expectedStride) {
            return false;
        }
        expectedStride *= shape[i];
    }

    return true;
}

/**
 * Naive global reduction (single workgroup)
 */
function executeNaiveGlobalReduction(iter: ITensorIterator, dispatchKey: string): void {
    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;
    const numel = iter.reductionNumel;
    const input = iter.input(0);

    // Check if input is contiguous
    const isContiguous = isInputContiguousForGlobalReduction(iter);
    const rank = iter.inputShape.length;

    logger.debug(`Naive global reduction: ${numel} elements, contiguous=${isContiguous}`);

    let shaderCode: string;
    let pipelineKey: string;

    if (isContiguous) {
        shaderCode = buildNaiveReductionShader(iter, dispatchKey, workgroupSize);
        pipelineKey = `reduction_naive.${dispatchKey}-${iter.computeDtype}-wg${workgroupSize}`;
    } else {
        shaderCode = buildStridedNaiveReductionShader(iter, dispatchKey, workgroupSize, rank);
        pipelineKey = `reduction_naive_strided.${dispatchKey}-${iter.computeDtype}-r${rank}-wg${workgroupSize}`;
    }

    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const module = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // Create uniform buffer
    let uniformBuffer: GPUBuffer;

    if (isContiguous) {
        // Contiguous path: simple uniforms
        const uniformData = new ArrayBuffer(16);
        const uniformView = new DataView(uniformData);
        uniformView.setUint32(0, numel, true);
        uniformView.setUint32(4, input.offset, true);
        uniformView.setUint32(8, iter.output().offset, true);
        uniformBuffer = createUniformBuffer(uniformData);
    } else {
        // Strided path: include shape and strides
        uniformBuffer = createStridedGlobalReductionUniformBuffer(device, iter, numel);
    }

    const inputBuffer = input.buffer as GPUBuffer;
    const outputBuffer = iter.output().buffer as GPUBuffer;

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: inputBuffer } },
            { binding: 2, resource: { buffer: outputBuffer } },
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

/**
 * Tree global reduction (multi-stage)
 */
function executeTreeGlobalReduction(iter: ITensorIterator, dispatchKey: string): void {
    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;
    const numel = iter.reductionNumel;
    const resolver = getGlobalDTypeResolver();
    const computeType = getComputeType(iter.computeDtype);
    const outputType = resolver.getDescriptor(iter.output().dtype).wgslStorageType;
    const input = iter.input(0);

    // Check if input is contiguous
    const isContiguous = isInputContiguousForGlobalReduction(iter);
    const rank = iter.inputShape.length;

    logger.debug(`Tree global reduction: ${numel} elements, contiguous=${isContiguous}`);

    // Stage 1: Reduce input to partial results
    const numWorkgroups = Math.ceil(numel / workgroupSize);

    let stage1Shader: string;
    let stage1Key: string;

    if (isContiguous) {
        stage1Shader = buildGlobalReductionStage1(iter, dispatchKey, workgroupSize);
        stage1Key = `reduction_s1.${dispatchKey}-${iter.computeDtype}-wg${workgroupSize}`;
    } else {
        stage1Shader = buildStridedGlobalReductionStage1(iter, dispatchKey, workgroupSize, rank);
        stage1Key = `reduction_s1_strided.${dispatchKey}-${iter.computeDtype}-r${rank}-wg${workgroupSize}`;
    }

    let stage1Pipeline = WebGPUPipelineManager.getPipeline(stage1Key);

    if (!stage1Pipeline) {
        const module = device.createShaderModule({ code: stage1Shader });
        stage1Pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(stage1Key, stage1Pipeline);
    }

    // Create partial results buffer
    const computeTypeSize = computeType === 'f32' ? 4 : (computeType === 'f16' ? 2 : 4);
    const partialResultsBuffer = device.createBuffer({
        size: numWorkgroups * computeTypeSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Stage 1 uniforms
    let stage1Uniforms: GPUBuffer;

    if (isContiguous) {
        const stage1Data = new ArrayBuffer(16);
        const stage1View = new DataView(stage1Data);
        stage1View.setUint32(0, numel, true);
        stage1View.setUint32(4, input.offset, true);
        stage1Uniforms = createUniformBuffer(stage1Data);
    } else {
        stage1Uniforms = createStridedGlobalReductionStage1UniformBuffer(device, iter, numel);
    }

    const inputBuffer = input.buffer as GPUBuffer;

    const stage1BindGroup = device.createBindGroup({
        layout: stage1Pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: stage1Uniforms } },
            { binding: 1, resource: { buffer: inputBuffer } },
            { binding: 2, resource: { buffer: partialResultsBuffer } },
        ],
    });

    const commandEncoder = device.createCommandEncoder();

    // Stage 1 dispatch
    const stage1Pass = commandEncoder.beginComputePass();
    stage1Pass.setPipeline(stage1Pipeline);
    stage1Pass.setBindGroup(0, stage1BindGroup);
    stage1Pass.dispatchWorkgroups(numWorkgroups);
    stage1Pass.end();

    // Stage 2: Reduce partial results to final result
    const stage2Shader = buildGlobalReductionStage2(dispatchKey, computeType, outputType, workgroupSize, numel);
    const stage2Key = `reduction_s2.${dispatchKey}-${iter.computeDtype}-${iter.output().dtype}-n${numel}-wg${workgroupSize}`;
    let stage2Pipeline = WebGPUPipelineManager.getPipeline(stage2Key);

    if (!stage2Pipeline) {
        const module = device.createShaderModule({ code: stage2Shader });
        stage2Pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(stage2Key, stage2Pipeline);
    }

    // Stage 2 uniforms
    const stage2Data = new ArrayBuffer(16);
    const stage2View = new DataView(stage2Data);
    stage2View.setUint32(0, numWorkgroups, true);
    stage2View.setUint32(4, iter.output().offset, true);
    const stage2Uniforms = createUniformBuffer(stage2Data);

    const outputBuffer = iter.output().buffer as GPUBuffer;

    const stage2BindGroup = device.createBindGroup({
        layout: stage2Pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: stage2Uniforms } },
            { binding: 1, resource: { buffer: partialResultsBuffer } },
            { binding: 2, resource: { buffer: outputBuffer } },
        ],
    });

    // Stage 2 dispatch
    const stage2Pass = commandEncoder.beginComputePass();
    stage2Pass.setPipeline(stage2Pipeline);
    stage2Pass.setBindGroup(0, stage2BindGroup);
    stage2Pass.dispatchWorkgroups(1);
    stage2Pass.end();

    device.queue.submit([commandEncoder.finish()]);

    logger.debug(`Tree global reduction complete: ${numWorkgroups} -> 1`);
}

/**
 * Create uniform buffer for strided naive global reduction
 * Layout: numel(4), rank(4), input_offset(4), output_offset(4), shape0(16), shape1(16), strides0(16), strides1(16)
 * Total: 80 bytes, aligned to 80
 */
function createStridedGlobalReductionUniformBuffer(device: GPUDevice, iter: ITensorIterator, numel: number): GPUBuffer {
    const input = iter.input(0);
    const shape = iter.inputShape;
    const strides = input.tensorHandle.strides;
    const rank = shape.length;

    // Layout: numel, rank, input_offset, output_offset (16), shape0/1 (32), strides0/1 (32) = 80 bytes
    const uniformSize = 80;
    const uniformData = new ArrayBuffer(uniformSize);
    const uniformView = new DataView(uniformData);
    let offset = 0;

    uniformView.setUint32(offset, numel, true); offset += 4;
    uniformView.setUint32(offset, rank, true); offset += 4;
    uniformView.setUint32(offset, input.offset, true); offset += 4;
    uniformView.setUint32(offset, iter.output().offset, true); offset += 4;

    // shape0 (vec4)
    for (let i = 0; i < 4; i++) {
        const dim = i < rank ? shape[i] : 1;
        uniformView.setUint32(offset, dim, true); offset += 4;
    }
    // shape1 (vec4)
    for (let i = 4; i < 8; i++) {
        const dim = i < rank ? shape[i] : 1;
        uniformView.setUint32(offset, dim, true); offset += 4;
    }
    // strides0 (vec4)
    for (let i = 0; i < 4; i++) {
        const stride = i < rank ? strides[i] : 0;
        uniformView.setUint32(offset, stride, true); offset += 4;
    }
    // strides1 (vec4)
    for (let i = 4; i < 8; i++) {
        const stride = i < rank ? strides[i] : 0;
        uniformView.setUint32(offset, stride, true); offset += 4;
    }

    return createUniformBuffer(uniformData);
}

/**
 * Create uniform buffer for strided tree global reduction Stage 1
 * Layout: numel(4), rank(4), input_offset(4), _pad(4), shape0(16), shape1(16), strides0(16), strides1(16)
 * Total: 80 bytes
 */
function createStridedGlobalReductionStage1UniformBuffer(device: GPUDevice, iter: ITensorIterator, numel: number): GPUBuffer {
    const input = iter.input(0);
    const shape = iter.inputShape;
    const strides = input.tensorHandle.strides;
    const rank = shape.length;

    const uniformSize = 80;
    const uniformData = new ArrayBuffer(uniformSize);
    const uniformView = new DataView(uniformData);
    let offset = 0;

    uniformView.setUint32(offset, numel, true); offset += 4;
    uniformView.setUint32(offset, rank, true); offset += 4;
    uniformView.setUint32(offset, input.offset, true); offset += 4;
    uniformView.setUint32(offset, 0, true); offset += 4; // _pad0

    // shape0 (vec4)
    for (let i = 0; i < 4; i++) {
        const dim = i < rank ? shape[i] : 1;
        uniformView.setUint32(offset, dim, true); offset += 4;
    }
    // shape1 (vec4)
    for (let i = 4; i < 8; i++) {
        const dim = i < rank ? shape[i] : 1;
        uniformView.setUint32(offset, dim, true); offset += 4;
    }
    // strides0 (vec4)
    for (let i = 0; i < 4; i++) {
        const stride = i < rank ? strides[i] : 0;
        uniformView.setUint32(offset, stride, true); offset += 4;
    }
    // strides1 (vec4)
    for (let i = 4; i < 8; i++) {
        const stride = i < rank ? strides[i] : 0;
        uniformView.setUint32(offset, stride, true); offset += 4;
    }

    return createUniformBuffer(uniformData);
}

/**
 * Create uniform buffer for dimensional reduction
 * Uses vec4 types for proper WGSL alignment
 * 
 * IMPORTANT: When keepdim=true, outputShape includes reduced dims (size=1).
 * But input.strides (parallelStrides) only contains non-reduced dims.
 * We use iter.reductionAxes to explicitly know which dimensions were reduced.
 */
function createDimReductionUniformBuffer(device: GPUDevice, iter: ITensorIterator): GPUBuffer {
    const input = iter.input(0);
    const output = iter.output();

    const outputRank = iter.outputShape.length;
    const parallelStrides = input.strides;  // These are only non-reduction dims
    // reductionStrides is properly defined in TensorIteratorOperand interface
    const reductionStrides = input.reductionStrides || [];
    const reductionRank = iter.reductionShape.length;

    // Use explicit reductionAxes from iter (no more stride===0 assumption!)
    const reductionAxesSet = new Set(iter.reductionAxes);

    // Reconstruct input_parallel_strides to match output dimensions
    let inputParallelStridesForOutput: number[];

    if (!iter.keepDims || outputRank === parallelStrides.length) {
        // keepdim=false or same rank: direct mapping
        inputParallelStridesForOutput = [...parallelStrides];
    } else {
        // keepdim=true case: outputShape has more dims than parallelStrides
        // Insert 0 for dimensions that were reduced (identified by reductionAxes)
        inputParallelStridesForOutput = [];
        let parallelIdx = 0;
        for (let i = 0; i < outputRank; i++) {
            if (reductionAxesSet.has(i)) {
                // This is a reduced dimension: stride = 0 (no contribution from this coord)
                inputParallelStridesForOutput.push(0);
            } else {
                // Parallel dimension: use next parallel stride
                inputParallelStridesForOutput.push(parallelStrides[parallelIdx] || 0);
                parallelIdx++;
            }
        }
    }

    // Layout (all vec4 aligned):
    // output_numel(4), reduction_numel(4), output_rank(4), reduction_rank(4) = 16 bytes
    // output_shape0(16), output_shape1(16) = 32 bytes
    // reduction_shape0(16), reduction_shape1(16) = 32 bytes
    // input_parallel_strides0(16), input_parallel_strides1(16) = 32 bytes
    // input_reduction_strides0(16), input_reduction_strides1(16) = 32 bytes
    // output_strides0(16), output_strides1(16) = 32 bytes
    // input_offset(4), output_offset(4) + padding = 16 bytes
    // Total: 192 bytes
    const uniformSize = 192;

    const uniformData = new ArrayBuffer(uniformSize);
    const uniformView = new DataView(uniformData);
    let offset = 0;

    // Header: output_numel, reduction_numel, output_rank, reduction_rank
    uniformView.setUint32(offset, iter.outputNumel, true); offset += 4;
    uniformView.setUint32(offset, iter.reductionNumel, true); offset += 4;
    uniformView.setUint32(offset, iter.outputShape.length, true); offset += 4;
    uniformView.setUint32(offset, iter.reductionShape.length, true); offset += 4;

    // output_shape0 (vec4)
    for (let i = 0; i < 4; i++) {
        const dim = i < iter.outputShape.length ? iter.outputShape[i] : 1;
        uniformView.setUint32(offset, dim, true); offset += 4;
    }
    // output_shape1 (vec4)
    for (let i = 4; i < 8; i++) {
        const dim = i < iter.outputShape.length ? iter.outputShape[i] : 1;
        uniformView.setUint32(offset, dim, true); offset += 4;
    }

    // reduction_shape0 (vec4)
    for (let i = 0; i < 4; i++) {
        const dim = i < iter.reductionShape.length ? iter.reductionShape[i] : 1;
        uniformView.setUint32(offset, dim, true); offset += 4;
    }
    // reduction_shape1 (vec4)
    for (let i = 4; i < 8; i++) {
        const dim = i < iter.reductionShape.length ? iter.reductionShape[i] : 1;
        uniformView.setUint32(offset, dim, true); offset += 4;
    }

    // input_parallel_strides - padded and aligned to output rank
    for (let i = 0; i < 4; i++) {
        const stride = i < inputParallelStridesForOutput.length ? inputParallelStridesForOutput[i] : 0;
        uniformView.setUint32(offset, stride, true); offset += 4;
    }
    for (let i = 4; i < 8; i++) {
        const stride = i < inputParallelStridesForOutput.length ? inputParallelStridesForOutput[i] : 0;
        uniformView.setUint32(offset, stride, true); offset += 4;
    }

    // input_reduction_strides
    for (let i = 0; i < 4; i++) {
        const stride = i < reductionStrides.length ? reductionStrides[i] : 0;
        uniformView.setUint32(offset, stride, true); offset += 4;
    }
    for (let i = 4; i < 8; i++) {
        const stride = i < reductionStrides.length ? reductionStrides[i] : 0;
        uniformView.setUint32(offset, stride, true); offset += 4;
    }

    // output_strides
    const outputStrides = output.strides;
    for (let i = 0; i < 4; i++) {
        const stride = i < outputStrides.length ? outputStrides[i] : 0;
        uniformView.setUint32(offset, stride, true); offset += 4;
    }
    for (let i = 4; i < 8; i++) {
        const stride = i < outputStrides.length ? outputStrides[i] : 0;
        uniformView.setUint32(offset, stride, true); offset += 4;
    }

    // input_offset, output_offset
    uniformView.setUint32(offset, input.offset, true); offset += 4;
    uniformView.setUint32(offset, output.offset, true); offset += 4;

    return createUniformBuffer(uniformData);
}
