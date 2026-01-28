/**
 * Sort Kernel Executor (v5)
 *
 * Unified executor for all sort operations: topk, sort, argsort
 * 
 * Architecture:
 * - Uses DirectContext pattern (like MatrixHandler, not TensorIterator)
 * - Registry-driven dispatch via SORT_OPS configuration
 * - Pipeline caching via WebGPUPipelineManager
 * - Multi-dtype support via DTypeResolver
 * - Supports strided tensors via shape/strides uniforms
 * 
 * The executor handles:
 * 1. Parameter extraction and validation
 * 2. Output tensor allocation
 * 3. Uniform buffer preparation
 * 4. Shader generation and pipeline caching
 * 5. GPU dispatch
 */

import type { ITensorHandle, DType } from '@kandle/types';
import { Logger, computeStrides } from '@kandle/utils';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { getGlobalDTypeResolver, type PhysicalStorageDescriptor } from '../../base/DTypeResolver';
import { WebGPUTensor } from '../../base/tensor';
import { SORT_OPS, getOutputDimSize, validateSortParams } from './ops';
import { buildSortShader, calculateBitonicStages } from './shaderBuilder';
import type { SortConfig, SortOpConfig, SortScalarArgs, SortOutputs, SortKernelImpl } from './types';
import { createUniformBuffer } from '../../base/uniformUtils';

const logger = new Logger('Sort-Executor');

// DEBUG: Print when module is loaded to verify fresh code
console.log('[Sort-Executor] Module loaded - v2 (2025-12-25 07:00)');

// ============================================================
// Main Entry Points
// ============================================================

/**
 * Execute topk operation
 * Returns [values, indices] tuple
 */
export const topkKernel: SortKernelImpl = (
    input: ITensorHandle,
    scalars: Record<string, unknown>,
    outs?: ITensorHandle[]
): [ITensorHandle, ITensorHandle] => {
    console.log('[topkKernel] Entry point called');
    return executeSort(input, scalars, 'topk', outs) as [ITensorHandle, ITensorHandle];
};

/**
 * Execute sort operation
 * Returns [values, indices] tuple
 */
export const sortKernel: SortKernelImpl = (
    input: ITensorHandle,
    scalars: Record<string, unknown>,
    outs?: ITensorHandle[]
): [ITensorHandle, ITensorHandle] => {
    console.log('[sortKernel] Entry point called');
    return executeSort(input, scalars, 'sort', outs) as [ITensorHandle, ITensorHandle];
};

/**
 * Execute argsort operation
 * Returns indices tensor only
 */
export const argsortKernel: SortKernelImpl = (
    input: ITensorHandle,
    scalars: Record<string, unknown>,
    outs?: ITensorHandle[]
): ITensorHandle => {
    console.log('[argsortKernel] Entry point called');
    return executeSort(input, scalars, 'argsort', outs) as ITensorHandle;
};

// ============================================================
// Unified Executor
// ============================================================

/**
 * Unified sort executor
 * 
 * Processes all sort operations through a common pipeline:
 * 1. Extract and validate configuration
 * 2. Allocate output tensors
 * 3. Execute on GPU
 * 4. Return appropriate results
 */
export function executeSort(
    input: ITensorHandle,
    scalars: Record<string, unknown>,
    dispatchKey: string,
    outs?: ITensorHandle[]
): ITensorHandle | [ITensorHandle, ITensorHandle] {
    const opConfig = SORT_OPS[dispatchKey];
    if (!opConfig) {
        throw new Error(`Unknown sort operation: ${dispatchKey}`);
    }

    // TEMPORARY DEBUG - remove after fixing
    console.log(`[Sort Kernel] Executing ${dispatchKey}: shape=[${input.shape}], dtype=${input.dtype}`);
    logger.debug(`Executing ${dispatchKey}: shape=${input.shape}, dtype=${input.dtype}`);

    // 1. Extract and validate configuration
    const config = extractSortConfig(input, scalars, opConfig, dispatchKey);
    validateSortConfig(config);

    // 2. Allocate output tensors
    const outputs = allocateOutputs(config, outs);

    // 3. Execute on GPU
    executeSortGPU(config, outputs);

    // 4. Return results based on operation type
    if (opConfig.returns === 'tuple') {
        return [outputs.values!, outputs.indices!];
    } else if (opConfig.returns === 'indices') {
        return outputs.indices!;
    } else {
        return outputs.values!;
    }
}

// ============================================================
// Configuration Processing
// ============================================================

/**
 * Extract configuration from raw parameters
 */
function extractSortConfig(
    input: ITensorHandle,
    scalars: Record<string, unknown>,
    opConfig: SortOpConfig,
    dispatchKey: string
): SortConfig {
    // Apply defaults and extract scalars
    const processedScalars: SortScalarArgs = {
        k: undefined,
        dim: -1,
        largest: true,
        sorted: true,
        descending: false,
        stable: false,
    };

    for (const key of opConfig.scalarParams) {
        const value = scalars[key];
        const defaultValue = opConfig.scalarDefaults[key];
        const finalValue = value !== undefined ? value : defaultValue;

        // Type-safe assignment based on key
        switch (key) {
            case 'k':
                processedScalars.k = finalValue as number | undefined;
                break;
            case 'dim':
                processedScalars.dim = (finalValue as number) ?? -1;
                break;
            case 'largest':
                processedScalars.largest = (finalValue as boolean) ?? true;
                break;
            case 'sorted':
                processedScalars.sorted = (finalValue as boolean) ?? true;
                break;
            case 'descending':
                processedScalars.descending = (finalValue as boolean) ?? false;
                break;
            case 'stable':
                processedScalars.stable = (finalValue as boolean) ?? false;
                break;
        }
    }

    // Normalize dimension (handle negative indexing)
    const rawDim = processedScalars.dim;
    const normDim = rawDim < 0 ? input.shape.length + rawDim : rawDim;

    // Calculate output shape and dimension sizes
    const dimSize = input.shape[normDim];
    const outputDimSize = getOutputDimSize(dispatchKey, dimSize, processedScalars.k);

    // Calculate output shape
    const outputShape = [...input.shape];
    outputShape[normDim] = outputDimSize;

    // Calculate number of independent slices
    // This is the product of all dimensions except the sort dimension
    const numSlices = input.shape.reduce((acc, size, i) =>
        i === normDim ? acc : acc * size, 1
    );

    logger.debug(`Config: dim=${normDim}, dimSize=${dimSize}, numSlices=${numSlices}, outputDimSize=${outputDimSize}, k=${processedScalars.k}, largest=${processedScalars.largest}, descending=${processedScalars.descending}`);

    return {
        input,
        opConfig,
        dispatchKey,
        scalars: processedScalars,
        dim: normDim,
        dimSize,
        numSlices,
        outputShape,
        outputDimSize,
        dtype: input.dtype,
    };
}

/**
 * Validate sort configuration
 */
function validateSortConfig(config: SortConfig): void {
    const { input, dim, dimSize, dispatchKey, scalars } = config;

    // Validate dimension
    if (dim < 0 || dim >= input.shape.length) {
        throw new Error(
            `Invalid dimension ${dim} for tensor with ${input.shape.length} dimensions`
        );
    }

    // Validate k for topk
    validateSortParams(dispatchKey, dim, dimSize, scalars.k);

    // Validate non-empty tensor
    if (input.numel === 0) {
        throw new Error('Sort operations on empty tensors not supported');
    }

    // Check supported dtypes
    const supportedDtypes: DType[] = [
        'float32', 'float64', 'float16',
        'int32', 'int64', 'int16', 'int8',
        'uint32', 'uint64', 'uint16', 'uint8',
    ];
    if (!supportedDtypes.includes(input.dtype)) {
        throw new Error(`Sort does not support dtype: ${input.dtype}`);
    }
}

// ============================================================
// Output Allocation
// ============================================================

/**
 * Allocate output tensors
 */
function allocateOutputs(
    config: SortConfig,
    providedOuts?: ITensorHandle[]
): SortOutputs {
    const { outputShape, input, opConfig } = config;
    const outputs: SortOutputs = {};

    if (opConfig.needsValues) {
        outputs.values = providedOuts?.[0] as WebGPUTensor<typeof input.dtype>
            ?? WebGPUTensor.createNew(outputShape as number[], input.dtype);
    }

    if (opConfig.needsIndices) {
        const indexTensorIdx = opConfig.needsValues ? 1 : 0;
        // PyTorch uses int64 for indices, but WebGPU uses i32 internally
        outputs.indices = providedOuts?.[indexTensorIdx] as WebGPUTensor<'int32'>
            ?? WebGPUTensor.createNew(outputShape as number[], 'int32');
    }

    return outputs;
}

// ============================================================
// GPU Execution
// ============================================================

/**
 * Execute sort on GPU
 */
function executeSortGPU(
    config: SortConfig,
    outputs: SortOutputs
): void {
    const device = WebGPUDeviceManager.device;
    const resolver = getGlobalDTypeResolver();

    // Get dtype descriptor
    const descriptor = resolver.getDescriptor(config.dtype);

    // Build pipeline key
    const pipelineKey = computeSortPipelineKey(config);

    // Try to get cached pipeline
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    // Create bind group layout if not exists
    const layoutKey = `sort.${config.dispatchKey}-layout`;
    let bindGroupLayout = WebGPUPipelineManager.getBindGroupLayout(layoutKey);

    if (!bindGroupLayout) {
        bindGroupLayout = createSortBindGroupLayout(device, config.opConfig);
        WebGPUPipelineManager.registerBindGroupLayout(layoutKey, bindGroupLayout);
    }

    if (!pipeline) {
        // Generate shader
        const shaderCode = buildSortShader(config, descriptor);
        logger.debug(`Generated shader for ${pipelineKey}`);

        // Create shader module with error checking
        const shaderModule = device.createShaderModule({ code: shaderCode });

        // Check for shader compilation errors
        shaderModule.getCompilationInfo().then(info => {
            for (const message of info.messages) {
                const msgType = message.type === 'error' ? 'ERROR' : message.type === 'warning' ? 'WARN' : 'INFO';
                console.error(`[Sort Shader ${msgType}] Line ${message.lineNum}:${message.linePos}: ${message.message}`);
                if (message.type === 'error') {
                    // Log the problematic section of shader code
                    const lines = shaderCode.split('\n');
                    const startLine = Math.max(0, message.lineNum - 3);
                    const endLine = Math.min(lines.length, message.lineNum + 2);
                    console.error('Shader context:');
                    for (let i = startLine; i < endLine; i++) {
                        const marker = i === message.lineNum - 1 ? '>>> ' : '    ';
                        console.error(`${marker}${i + 1}: ${lines[i]}`);
                    }
                }
            }
        });

        pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module: shaderModule, entryPoint: 'main' },
        });

        // Cache pipeline
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // Create uniform buffer
    const uniformBuffer = createSortUniformBuffer(device, config, outputs, descriptor);

    // Get input buffer
    const inputTensor = config.input as WebGPUTensor<typeof config.dtype>;
    const inputBuffer = inputTensor.buffer;

    // Create bind group
    const bindGroupEntries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: inputBuffer } },
    ];

    let bindingIdx = 2;
    if (outputs.values) {
        const valuesTensor = outputs.values as WebGPUTensor<typeof config.dtype>;
        bindGroupEntries.push({
            binding: bindingIdx++,
            resource: { buffer: valuesTensor.buffer }
        });
    }
    if (outputs.indices) {
        const indicesTensor = outputs.indices as WebGPUTensor<'int32'>;
        bindGroupEntries.push({
            binding: bindingIdx,
            resource: { buffer: indicesTensor.buffer }
        });
    }

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: bindGroupEntries,
    });

    // Dispatch: one workgroup per slice
    const numWorkgroups = config.numSlices;
    console.log(`[Sort Kernel] Dispatching ${numWorkgroups} workgroups for ${config.dispatchKey}`);
    logger.debug(`Dispatching ${numWorkgroups} workgroups`);

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
    console.log(`[Sort Kernel] GPU commands submitted for ${config.dispatchKey}`);
}

/**
 * Compute pipeline cache key
 */
function computeSortPipelineKey(config: SortConfig): string {
    const { dispatchKey, dtype, dimSize, outputDimSize, opConfig, scalars } = config;

    // Include factors that affect shader generation
    const paddedSize = nextPowerOf2(dimSize);

    return `sort.${dispatchKey}.${dtype}.${opConfig.algorithm}.d${dimSize}.p${paddedSize}.o${outputDimSize}`;
}

/**
 * Create bind group layout for sort operations
 */
function createSortBindGroupLayout(
    device: GPUDevice,
    opConfig: SortOpConfig
): GPUBindGroupLayout {
    const entries: GPUBindGroupLayoutEntry[] = [
        // Uniform buffer
        {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'uniform' }
        },
        // Input buffer
        {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'read-only-storage' }
        },
    ];

    let bindingIdx = 2;
    if (opConfig.needsValues) {
        entries.push({
            binding: bindingIdx++,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'storage' }
        });
    }
    if (opConfig.needsIndices) {
        entries.push({
            binding: bindingIdx,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'storage' }
        });
    }

    return device.createBindGroupLayout({ entries });
}

/**
 * Create uniform buffer for sort operation
 * 
 * Layout matches buildUniformsStruct in shaderBuilder.ts
 */
function createSortUniformBuffer(
    device: GPUDevice,
    config: SortConfig,
    outputs: SortOutputs,
    descriptor: PhysicalStorageDescriptor
): GPUBuffer {
    const { input, dim, dimSize, numSlices, outputDimSize, scalars } = config;

    // Buffer layout:
    // Header: 16 bytes
    // Input shape: 32 bytes
    // Input strides: 32 bytes
    // Output strides: 32 bytes
    // Offsets + rank: 16 bytes
    // Scalars: 16 bytes
    const uniformSize = 16 + 32 + 32 + 32 + 16 + 16;  // 144 bytes

    const uniformData = new ArrayBuffer(uniformSize);
    const view = new DataView(uniformData);
    let offset = 0;

    // Header (16 bytes)
    view.setUint32(offset, dimSize, true); offset += 4;
    view.setUint32(offset, numSlices, true); offset += 4;
    view.setUint32(offset, dim, true); offset += 4;
    view.setUint32(offset, outputDimSize, true); offset += 4;

    // Input shape (32 bytes)
    for (let i = 0; i < 8; i++) {
        const dimVal = i < input.shape.length ? input.shape[i] : 1;
        view.setUint32(offset, dimVal, true);
        offset += 4;
    }

    // Input strides (32 bytes)
    for (let i = 0; i < 8; i++) {
        const stride = i < input.strides.length ? input.strides[i] : 0;
        view.setUint32(offset, stride, true);
        offset += 4;
    }

    // Output strides (32 bytes)
    const outputShape = [...input.shape];
    outputShape[dim] = outputDimSize;
    const outputStrides = computeStrides(outputShape);
    logger.debug(`Output shape: [${outputShape.join(', ')}], strides: [${outputStrides.join(', ')}]`);
    for (let i = 0; i < 8; i++) {
        const stride = i < outputStrides.length ? outputStrides[i] : 0;
        view.setUint32(offset, stride, true);
        offset += 4;
    }

    // Offsets + rank (16 bytes)
    const bytesPerElement = descriptor.gpuBytesPerElement;
    const inputOffset = input.offset / bytesPerElement;
    const valuesOffset = outputs.values
        ? (outputs.values as WebGPUTensor<typeof config.dtype>).offset / bytesPerElement
        : 0;
    const indicesOffset = outputs.indices
        ? (outputs.indices as WebGPUTensor<'int32'>).offset / 4  // i32 = 4 bytes
        : 0;

    view.setUint32(offset, inputOffset, true); offset += 4;
    view.setUint32(offset, valuesOffset, true); offset += 4;
    view.setUint32(offset, indicesOffset, true); offset += 4;
    view.setUint32(offset, input.shape.length, true); offset += 4;

    // Scalars (16 bytes)
    // Compute descending flag based on operation type:
    // - topk: largest=true means we want biggest values first → descending sort
    // - sort/argsort: descending parameter directly controls sort direction
    // 
    // Note: scalars.descending is set for sort/argsort, scalars.largest is set for topk
    // For sort/argsort: descending defaults to false, largest defaults to true
    // The unified logic: isDescending = descending OR largest (for topk with largest=true)
    const isDescending = config.dispatchKey === 'topk'
        ? scalars.largest   // topk: largest=true → descending (big values first)
        : scalars.descending;  // sort/argsort: use descending param directly
    const descendingFlag = isDescending ? 1 : 0;
    logger.debug(`Sort direction: isDescending=${isDescending}, descendingFlag=${descendingFlag}`);
    view.setUint32(offset, descendingFlag, true); offset += 4;
    // sorted flag
    view.setUint32(offset, scalars.sorted ? 1 : 0, true); offset += 4;
    // padding
    view.setUint32(offset, 0, true); offset += 4;
    view.setUint32(offset, 0, true); offset += 4;

    return createUniformBuffer(uniformData);
}

/**
 * Calculate next power of 2 >= n
 */
function nextPowerOf2(n: number): number {
    if (n <= 1) return 1;
    let p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}
