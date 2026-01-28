/**
 * Scan Kernel Executor (v5)
 * 
 * Unified scan executor using SCAN_OPS registry
 * 
 * Supports:
 * - cumsum, cumprod: Single output prefix sum/product
 * - cummax, cummin: Dual output (values + indices)
 * 
 * Uses Blelloch work-efficient parallel scan algorithm:
 * - Single-pass for small dimensions (scanDimSize <= 1024)
 * - Multi-pass for large dimensions (up-sweep, down-sweep, block add)
 */

import { ITensorIterator, DType } from '@kandle/types';
import { Logger } from '@kandle/utils';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { getGlobalDTypeResolver } from '../../base/DTypeResolver';
import { getComputeType } from '../../base/dtype';
import { SCAN_OPS } from './ops';
import { ScanDimParams, SCAN_SINGLE_PASS_THRESHOLD } from './types';
import {
    buildSinglePassScanShader,
    buildMultiPassScanStage1,
    buildMultiPassScanStage2,
    buildMultiPassScanStage3,
    buildCumExtremumShader
} from './shaderBuilder';
import { createUniformBuffer } from '../../base/uniformUtils';

const logger = new Logger('Scan-Executor');

/**
 * Main scan executor entry point
 */
export function executeScan(iter: ITensorIterator, dispatchKey: string): void {
    const opConfig = SCAN_OPS[dispatchKey];
    if (!opConfig) {
        throw new Error(`Unknown scan operation: ${dispatchKey}`);
    }

    // Get scan dimension from iterator's scalar args
    // Cast to access getScalarArg() which exists on TensorIterator implementation but not in ITensorIterator interface
    const scanDimRaw = (iter as any).getScalarArg?.('dim') as number | undefined;
    if (scanDimRaw === undefined) {
        throw new Error(`Scan operation requires 'dim' parameter`);
    }

    // Normalize dimension (handle negative)
    const rank = iter.inputShape.length;
    const scanDim = scanDimRaw < 0 ? scanDimRaw + rank : scanDimRaw;

    if (scanDim < 0 || scanDim >= rank) {
        throw new Error(`Invalid scan dimension ${scanDimRaw} for tensor of rank ${rank}`);
    }

    // Parse dimension information
    const params = parseScanParams(iter, scanDim);

    logger.debug(`Scan ${dispatchKey}: dim=${scanDim}, size=${params.scanDimSize}, outer=${params.outerSize}, inner=${params.innerSize}`);

    // Route to appropriate implementation
    if (opConfig.hasIndices) {
        // cummax/cummin with indices output
        executeCumExtremum(iter, dispatchKey, params);
    } else if (params.scanDimSize <= SCAN_SINGLE_PASS_THRESHOLD) {
        // Small dimension: single-pass scan
        executeSinglePassScan(iter, dispatchKey, params);
    } else {
        // Large dimension: multi-pass scan
        executeMultiPassScan(iter, dispatchKey, params);
    }
}

/**
 * Parse scan dimension parameters from iterator
 */
function parseScanParams(iter: ITensorIterator, scanDim: number): ScanDimParams {
    const inputShape = iter.inputShape;
    const input = iter.input(0);
    const inputStrides = input.tensorHandle.strides;

    const scanDimSize = inputShape[scanDim];

    // Compute outer size (product of dims before scanDim)
    let outerSize = 1;
    for (let i = 0; i < scanDim; i++) {
        outerSize *= inputShape[i];
    }

    // Compute inner size (product of dims after scanDim)
    let innerSize = 1;
    for (let i = scanDim + 1; i < inputShape.length; i++) {
        innerSize *= inputShape[i];
    }

    return {
        scanDim,
        scanDimSize,
        outerSize,
        innerSize,
        inputShape,
        inputStrides,
    };
}

/**
 * Execute single-pass scan for small dimensions
 */
function executeSinglePassScan(
    iter: ITensorIterator,
    dispatchKey: string,
    params: ScanDimParams
): void {
    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;

    const { scanDimSize, outerSize, innerSize, inputStrides, scanDim } = params;
    const totalSlices = outerSize * innerSize;

    logger.debug(`Single-pass scan: ${totalSlices} slices, ${scanDimSize} elements each`);

    // Generate shader
    const shaderCode = buildSinglePassScanShader(iter, dispatchKey, workgroupSize, params);

    // Pipeline key
    const pipelineKey = `scan_single.${dispatchKey}-${iter.computeDtype}-s${scanDimSize}-wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const module = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
        logger.debug(`Created single-pass scan pipeline: ${pipelineKey}`);
    }

    // Create uniform buffer
    const uniformBuffer = createSinglePassUniformBuffer(device, iter, params);

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

    // Dispatch: one workgroup per slice
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(totalSlices);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    logger.debug(`Single-pass scan complete: ${totalSlices} workgroups`);
}

/**
 * Execute multi-pass scan for large dimensions
 */
function executeMultiPassScan(
    iter: ITensorIterator,
    dispatchKey: string,
    params: ScanDimParams
): void {
    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;

    const { scanDimSize, outerSize, innerSize } = params;
    const totalSlices = outerSize * innerSize;

    // Elements per block = workgroupSize * 2 (Blelloch processes 2 elements per thread)
    const elementsPerBlock = workgroupSize * 2;
    const numBlocks = Math.ceil(scanDimSize / elementsPerBlock);
    const totalBlocks = totalSlices * numBlocks;

    logger.debug(`Multi-pass scan: ${totalSlices} slices, ${numBlocks} blocks per slice`);

    const resolver = getGlobalDTypeResolver();
    const computeType = getComputeType(iter.computeDtype);
    const computeTypeSize = computeType === 'f32' ? 4 : (computeType === 'f16' ? 2 : 4);
    const input = iter.input(0);
    const output = iter.output();

    // =========================================================================
    // Stage 1: Scan blocks and output block sums
    // =========================================================================

    const stage1Shader = buildMultiPassScanStage1(iter, dispatchKey, workgroupSize, params, elementsPerBlock);
    const stage1Key = `scan_s1.${dispatchKey}-${iter.computeDtype}-e${elementsPerBlock}-wg${workgroupSize}`;
    let stage1Pipeline = WebGPUPipelineManager.getPipeline(stage1Key);

    if (!stage1Pipeline) {
        const module = device.createShaderModule({ code: stage1Shader });
        stage1Pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(stage1Key, stage1Pipeline);
    }

    // Create temporary buffers
    // scanned_blocks: same size as output, holds per-block exclusive scan results
    const scannedBlocksSize = iter.outputNumel * getElementSize(output.dtype);
    const scannedBlocksBuffer = device.createBuffer({
        size: scannedBlocksSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // block_sums: one element per block across all slices
    const blockSumsBuffer = device.createBuffer({
        size: totalBlocks * computeTypeSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // block_prefixes: exclusive scan of block_sums
    const blockPrefixesBuffer = device.createBuffer({
        size: totalBlocks * computeTypeSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Stage 1 uniforms
    const stage1Uniforms = createMultiPassStage1UniformBuffer(device, iter, params, numBlocks, elementsPerBlock);

    const inputBuffer = input.buffer as GPUBuffer;

    const stage1BindGroup = device.createBindGroup({
        layout: stage1Pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: stage1Uniforms } },
            { binding: 1, resource: { buffer: inputBuffer } },
            { binding: 2, resource: { buffer: scannedBlocksBuffer } },
            { binding: 3, resource: { buffer: blockSumsBuffer } },
        ],
    });

    // =========================================================================
    // Stage 2: Scan block sums
    // =========================================================================

    const stage2Shader = buildMultiPassScanStage2(dispatchKey, computeType, workgroupSize, numBlocks, totalSlices);
    const stage2Key = `scan_s2.${dispatchKey}-${computeType}-nb${numBlocks}-wg${workgroupSize}`;
    let stage2Pipeline = WebGPUPipelineManager.getPipeline(stage2Key);

    if (!stage2Pipeline) {
        const module = device.createShaderModule({ code: stage2Shader });
        stage2Pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(stage2Key, stage2Pipeline);
    }

    const stage2Data = new ArrayBuffer(16);
    const stage2View = new DataView(stage2Data);
    stage2View.setUint32(0, numBlocks, true);
    stage2View.setUint32(4, totalSlices, true);
    const stage2Uniforms = createUniformBuffer(stage2Data);

    const stage2BindGroup = device.createBindGroup({
        layout: stage2Pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: stage2Uniforms } },
            { binding: 1, resource: { buffer: blockSumsBuffer } },
            { binding: 2, resource: { buffer: blockPrefixesBuffer } },
        ],
    });

    // =========================================================================
    // Stage 3: Add block prefixes and convert to inclusive
    // =========================================================================

    const stage3Shader = buildMultiPassScanStage3(iter, dispatchKey, workgroupSize, params, elementsPerBlock);
    const stage3Key = `scan_s3.${dispatchKey}-${iter.computeDtype}-e${elementsPerBlock}-wg${workgroupSize}`;
    let stage3Pipeline = WebGPUPipelineManager.getPipeline(stage3Key);

    if (!stage3Pipeline) {
        const module = device.createShaderModule({ code: stage3Shader });
        stage3Pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(stage3Key, stage3Pipeline);
    }

    const stage3Uniforms = createMultiPassStage3UniformBuffer(device, iter, params, numBlocks, elementsPerBlock);
    const outputBuffer = output.buffer as GPUBuffer;

    const stage3BindGroup = device.createBindGroup({
        layout: stage3Pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: stage3Uniforms } },
            { binding: 1, resource: { buffer: inputBuffer } },
            { binding: 2, resource: { buffer: scannedBlocksBuffer } },
            { binding: 3, resource: { buffer: blockPrefixesBuffer } },
            { binding: 4, resource: { buffer: outputBuffer } },
        ],
    });

    // =========================================================================
    // Execute all stages
    // =========================================================================

    const commandEncoder = device.createCommandEncoder();

    // Stage 1
    const stage1Pass = commandEncoder.beginComputePass();
    stage1Pass.setPipeline(stage1Pipeline);
    stage1Pass.setBindGroup(0, stage1BindGroup);
    stage1Pass.dispatchWorkgroups(totalBlocks);
    stage1Pass.end();

    // Stage 2
    const stage2Pass = commandEncoder.beginComputePass();
    stage2Pass.setPipeline(stage2Pipeline);
    stage2Pass.setBindGroup(0, stage2BindGroup);
    stage2Pass.dispatchWorkgroups(totalSlices);
    stage2Pass.end();

    // Stage 3
    const numStage3Workgroups = Math.ceil((totalSlices * scanDimSize) / workgroupSize);
    const stage3Pass = commandEncoder.beginComputePass();
    stage3Pass.setPipeline(stage3Pipeline);
    stage3Pass.setBindGroup(0, stage3BindGroup);
    stage3Pass.dispatchWorkgroups(numStage3Workgroups);
    stage3Pass.end();

    device.queue.submit([commandEncoder.finish()]);

    logger.debug(`Multi-pass scan complete: ${totalBlocks} blocks -> ${totalSlices} slices -> ${iter.outputNumel} elements`);
}

/**
 * Execute cummax/cummin with indices output
 */
function executeCumExtremum(
    iter: ITensorIterator,
    dispatchKey: string,
    params: ScanDimParams
): void {
    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;

    const { outerSize, innerSize } = params;
    const totalSlices = outerSize * innerSize;

    logger.debug(`Cumulative extremum ${dispatchKey}: ${totalSlices} slices`);

    // Generate shader
    const shaderCode = buildCumExtremumShader(iter, dispatchKey, workgroupSize, params);

    // Pipeline key
    const pipelineKey = `scan_ext.${dispatchKey}-${iter.computeDtype}-wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const module = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
        logger.debug(`Created cumulative extremum pipeline: ${pipelineKey}`);
    }

    // Create uniform buffer
    const uniformBuffer = createCumExtremumUniformBuffer(device, iter, params);

    // Get buffers
    const inputBuffer = iter.input(0).buffer as GPUBuffer;
    const valuesBuffer = iter.output(0).buffer as GPUBuffer;
    const indicesBuffer = iter.output(1).buffer as GPUBuffer;

    // Create bind group
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: inputBuffer } },
            { binding: 2, resource: { buffer: valuesBuffer } },
            { binding: 3, resource: { buffer: indicesBuffer } },
        ],
    });

    // Dispatch: one thread per slice
    const numWorkgroups = Math.ceil(totalSlices / workgroupSize);
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    logger.debug(`Cumulative extremum complete: ${numWorkgroups} workgroups`);
}

// ============================================================================
// Uniform Buffer Helpers
// ============================================================================

function createSinglePassUniformBuffer(
    device: GPUDevice,
    iter: ITensorIterator,
    params: ScanDimParams
): GPUBuffer {
    const input = iter.input(0);
    const output = iter.output();
    const { scanDimSize, outerSize, innerSize, inputStrides, scanDim } = params;

    // Calculate strides
    const scanDimStride = inputStrides[scanDim];
    const innerStride = params.inputShape.length > scanDim + 1 ? inputStrides[params.inputShape.length - 1] : 1;

    const uniformSize = 32; // 8 u32 values
    const uniformData = new ArrayBuffer(uniformSize);
    const uniformView = new DataView(uniformData);
    let offset = 0;

    uniformView.setUint32(offset, scanDimSize, true); offset += 4;
    uniformView.setUint32(offset, outerSize, true); offset += 4;
    uniformView.setUint32(offset, innerSize, true); offset += 4;
    uniformView.setUint32(offset, scanDimStride, true); offset += 4;
    uniformView.setUint32(offset, innerStride, true); offset += 4;
    uniformView.setUint32(offset, input.offset, true); offset += 4;
    uniformView.setUint32(offset, output.offset, true); offset += 4;
    uniformView.setUint32(offset, 0, true); // _pad

    return createUniformBuffer(uniformData);
}

function createMultiPassStage1UniformBuffer(
    device: GPUDevice,
    iter: ITensorIterator,
    params: ScanDimParams,
    numBlocks: number,
    elementsPerBlock: number
): GPUBuffer {
    const input = iter.input(0);
    const { scanDimSize, outerSize, innerSize, inputStrides, scanDim } = params;

    const scanDimStride = inputStrides[scanDim];
    const innerStride = params.inputShape.length > scanDim + 1 ? inputStrides[params.inputShape.length - 1] : 1;

    const uniformSize = 32;
    const uniformData = new ArrayBuffer(uniformSize);
    const uniformView = new DataView(uniformData);
    let offset = 0;

    uniformView.setUint32(offset, scanDimSize, true); offset += 4;
    uniformView.setUint32(offset, numBlocks, true); offset += 4;
    uniformView.setUint32(offset, elementsPerBlock, true); offset += 4;
    uniformView.setUint32(offset, outerSize, true); offset += 4;
    uniformView.setUint32(offset, innerSize, true); offset += 4;
    uniformView.setUint32(offset, scanDimStride, true); offset += 4;
    uniformView.setUint32(offset, innerStride, true); offset += 4;
    uniformView.setUint32(offset, input.offset, true);

    return createUniformBuffer(uniformData);
}

function createMultiPassStage3UniformBuffer(
    device: GPUDevice,
    iter: ITensorIterator,
    params: ScanDimParams,
    numBlocks: number,
    elementsPerBlock: number
): GPUBuffer {
    const input = iter.input(0);
    const output = iter.output();
    const { scanDimSize, outerSize, innerSize, inputStrides, scanDim } = params;

    const scanDimStride = inputStrides[scanDim];
    const innerStride = params.inputShape.length > scanDim + 1 ? inputStrides[params.inputShape.length - 1] : 1;

    const uniformSize = 48; // 10 u32 values, padded to multiple of 16
    const uniformData = new ArrayBuffer(uniformSize);
    const uniformView = new DataView(uniformData);
    let offset = 0;

    uniformView.setUint32(offset, scanDimSize, true); offset += 4;
    uniformView.setUint32(offset, numBlocks, true); offset += 4;
    uniformView.setUint32(offset, elementsPerBlock, true); offset += 4;
    uniformView.setUint32(offset, outerSize, true); offset += 4;
    uniformView.setUint32(offset, innerSize, true); offset += 4;
    uniformView.setUint32(offset, scanDimStride, true); offset += 4;
    uniformView.setUint32(offset, innerStride, true); offset += 4;
    uniformView.setUint32(offset, input.offset, true); offset += 4;
    uniformView.setUint32(offset, output.offset, true); offset += 4;
    uniformView.setUint32(offset, 0, true); // _pad

    return createUniformBuffer(uniformData);
}

function createCumExtremumUniformBuffer(
    device: GPUDevice,
    iter: ITensorIterator,
    params: ScanDimParams
): GPUBuffer {
    const input = iter.input(0);
    const valuesOutput = iter.output(0);
    const indicesOutput = iter.output(1);

    const { scanDimSize, outerSize, innerSize, inputStrides, scanDim } = params;

    const scanDimStride = inputStrides[scanDim];
    const innerStride = params.inputShape.length > scanDim + 1 ? inputStrides[params.inputShape.length - 1] : 1;

    const uniformSize = 32;
    const uniformData = new ArrayBuffer(uniformSize);
    const uniformView = new DataView(uniformData);
    let offset = 0;

    uniformView.setUint32(offset, scanDimSize, true); offset += 4;
    uniformView.setUint32(offset, outerSize, true); offset += 4;
    uniformView.setUint32(offset, innerSize, true); offset += 4;
    uniformView.setUint32(offset, scanDimStride, true); offset += 4;
    uniformView.setUint32(offset, innerStride, true); offset += 4;
    uniformView.setUint32(offset, input.offset, true); offset += 4;
    uniformView.setUint32(offset, valuesOutput.offset, true); offset += 4;
    uniformView.setUint32(offset, indicesOutput.offset, true);

    return createUniformBuffer(uniformData);
}

/**
 * Get element size in bytes for a dtype
 */
function getElementSize(dtype: DType): number {
    const sizeMap: Record<DType, number> = {
        'bool': 4,
        'int8': 1,
        'int16': 2,
        'int32': 4,
        'int64': 8,
        'uint8': 1,
        'uint16': 2,
        'uint32': 4,
        'uint64': 8,
        'float16': 2,
        'float32': 4,
        'float64': 8,
        'complex64': 8,
        'complex128': 16,
    };
    return sizeMap[dtype] || 4;
}
