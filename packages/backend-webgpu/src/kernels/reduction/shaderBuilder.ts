/**
 * Reduction Shader Builder (v5)
 * 
 * Generates WGSL shaders for reduction operations
 * Supports:
 * - Global reduction (all elements to scalar)
 * - Dimensional reduction (reduce along specific axes)
 * 
 * Uses REDUCTION_OPS registry for operator-specific code
 */

import { ITensorIterator } from '@kandle/types';
import { Logger } from '@kandle/utils';
import { getStorageType, generateLoadSnippet } from '../../shader/ShaderSnippets';
import { getComputeType, generateCastSnippet } from '../../base/dtype';
import { getGlobalDTypeResolver } from '../../base/DTypeResolver';
import { REDUCTION_OPS } from './ops';

const logger = new Logger('Reduction-ShaderBuilder');

/**
 * Build reduction shader for dimensional reduction
 * Each workgroup handles one output element, reducing over the reduction dimensions
 */
export function buildDimReductionShader(
    iter: ITensorIterator,
    dispatchKey: string,
    workgroupSize: number
): string {
    const opConfig = REDUCTION_OPS[dispatchKey];
    if (!opConfig) {
        throw new Error(`Unknown reduction operation: ${dispatchKey}`);
    }

    const input = iter.input(0);
    const output = iter.output();
    const resolver = getGlobalDTypeResolver();
    const computeType = getComputeType(iter.computeDtype);
    const storageTypeInput = getStorageType(input.dtype);
    const storageTypeOutput = getStorageType(output.dtype);
    const elemTypeInput = resolver.getDescriptor(input.dtype).wgslStorageType;
    const elemTypeOutput = resolver.getDescriptor(output.dtype).wgslStorageType;

    const loaderInput = generateLoadSnippet('input', input.dtype);

    const outputRank = iter.outputShape.length;
    const reductionRank = iter.reductionShape.length;

    // Generate tree reduction loop for workgroup
    const numSteps = Math.log2(workgroupSize);
    let treeReductionCode = '';
    for (let step = 0; step < numSteps; step++) {
        const stride = workgroupSize >> (step + 1);
        const accumulateCode = opConfig.accumulator('shared_data[tid]', `shared_data[tid + ${stride}u]`, computeType);
        treeReductionCode += `
        workgroupBarrier();
        if (tid < ${stride}u) {
            ${accumulateCode}
        }
        `;
    }

    const needsF16 = input.dtype === 'float16';
    const enableF16 = needsF16 && resolver.supportsNativeF16 ? 'enable f16;\n' : '';

    // Finalization for mean
    const needsFinalizer = opConfig.finalizer !== undefined;

    return `
${enableF16}
// Dimensional Reduction Shader: ${dispatchKey}
// Output shape: [${iter.outputShape.join(', ')}], Reduction shape: [${iter.reductionShape.join(', ')}]

struct Uniforms {
    output_numel: u32,
    reduction_numel: u32,
    output_rank: u32,
    reduction_rank: u32,
    // Output shape (vec4 * 2 = 8 dims max)
    output_shape0: vec4<u32>,
    output_shape1: vec4<u32>,
    // Reduction shape
    reduction_shape0: vec4<u32>,
    reduction_shape1: vec4<u32>,
    // Input parallel strides (for output dimensions)
    input_parallel_strides0: vec4<u32>,
    input_parallel_strides1: vec4<u32>,
    // Input reduction strides
    input_reduction_strides0: vec4<u32>,
    input_reduction_strides1: vec4<u32>,
    // Output strides
    output_strides0: vec4<u32>,
    output_strides1: vec4<u32>,
    // Offsets
    input_offset: u32,
    output_offset: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: ${storageTypeInput};
@group(0) @binding(2) var<storage, read_write> output: ${storageTypeOutput};

var<workgroup> shared_data: array<${computeType}, ${workgroupSize}>;

${loaderInput.code}

fn get_output_shape(dim: u32) -> u32 {
    if (dim < 4u) { return uniforms.output_shape0[dim]; }
    else { return uniforms.output_shape1[dim - 4u]; }
}

fn get_reduction_shape(dim: u32) -> u32 {
    if (dim < 4u) { return uniforms.reduction_shape0[dim]; }
    else { return uniforms.reduction_shape1[dim - 4u]; }
}

fn get_input_parallel_stride(dim: u32) -> u32 {
    if (dim < 4u) { return uniforms.input_parallel_strides0[dim]; }
    else { return uniforms.input_parallel_strides1[dim - 4u]; }
}

fn get_input_reduction_stride(dim: u32) -> u32 {
    if (dim < 4u) { return uniforms.input_reduction_strides0[dim]; }
    else { return uniforms.input_reduction_strides1[dim - 4u]; }
}

fn get_output_stride(dim: u32) -> u32 {
    if (dim < 4u) { return uniforms.output_strides0[dim]; }
    else { return uniforms.output_strides1[dim - 4u]; }
}

// Compute input base offset from output index (parallel dimensions)
fn compute_parallel_offset(output_flat_idx: u32) -> u32 {
    var offset: u32 = 0u;
    var remaining = output_flat_idx;
    for (var d: i32 = i32(uniforms.output_rank) - 1; d >= 0; d = d - 1) {
        let dim = u32(d);
        let dim_size = get_output_shape(dim);
        let coord = remaining % dim_size;
        remaining = remaining / dim_size;
        offset = offset + coord * get_input_parallel_stride(dim);
    }
    return offset;
}

// Compute input offset from reduction index
fn compute_reduction_offset(reduction_flat_idx: u32) -> u32 {
    var offset: u32 = 0u;
    var remaining = reduction_flat_idx;
    for (var d: i32 = i32(uniforms.reduction_rank) - 1; d >= 0; d = d - 1) {
        let dim = u32(d);
        let dim_size = get_reduction_shape(dim);
        let coord = remaining % dim_size;
        remaining = remaining / dim_size;
        offset = offset + coord * get_input_reduction_stride(dim);
    }
    return offset;
}

// Compute output offset from output index
fn compute_output_offset(output_flat_idx: u32) -> u32 {
    var offset: u32 = 0u;
    var remaining = output_flat_idx;
    for (var d: i32 = i32(uniforms.output_rank) - 1; d >= 0; d = d - 1) {
        let dim = u32(d);
        let dim_size = get_output_shape(dim);
        let coord = remaining % dim_size;
        remaining = remaining / dim_size;
        offset = offset + coord * get_output_stride(dim);
    }
    return offset;
}

@compute @workgroup_size(${workgroupSize})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let output_idx = wid.x;  // Each workgroup handles one output element
    
    if (output_idx >= uniforms.output_numel) {
        return;
    }
    
    // Compute base offset for this output element (parallel dimensions)
    let parallel_base = compute_parallel_offset(output_idx);
    
    // Initialize accumulator
    var acc = ${opConfig.initializer(computeType)};
    
    // Each thread processes multiple reduction elements (strided)
    for (var r = tid; r < uniforms.reduction_numel; r = r + ${workgroupSize}u) {
        let reduction_offset = compute_reduction_offset(r);
        let input_idx = uniforms.input_offset + parallel_base + reduction_offset;
        let raw_val = ${loaderInput.funcName}(input_idx);
        let val = ${generateCastSnippet('raw_val', elemTypeInput, computeType)};
        ${opConfig.accumulator('acc', 'val', computeType)};
    }
    
    shared_data[tid] = acc;
    
    // Tree reduction in shared memory
    ${treeReductionCode}
    
    // Thread 0 writes result
    if (tid == 0u) {
        var result = shared_data[0];
        ${needsFinalizer ? `result = ${opConfig.finalizer!('result', 'uniforms.reduction_numel', computeType)};` : ''}
        let out_idx = uniforms.output_offset + compute_output_offset(output_idx);
        output[out_idx] = ${computeType === elemTypeOutput ? 'result' : generateCastSnippet('result', computeType as any, elemTypeOutput)};
    }
}
`;
}

/**
 * Build global reduction Stage 1 shader for CONTIGUOUS inputs
 * All elements reduce to partial results
 */
export function buildGlobalReductionStage1(
    iter: ITensorIterator,
    dispatchKey: string,
    workgroupSize: number
): string {
    const opConfig = REDUCTION_OPS[dispatchKey];
    if (!opConfig) {
        throw new Error(`Unknown reduction operation: ${dispatchKey}`);
    }

    const input = iter.input(0);
    const resolver = getGlobalDTypeResolver();
    const computeType = getComputeType(iter.computeDtype);
    const storageTypeInput = getStorageType(input.dtype);
    const elemTypeInput = resolver.getDescriptor(input.dtype).wgslStorageType;

    const loaderInput = generateLoadSnippet('input', input.dtype);

    // Generate tree reduction loop
    const numSteps = Math.log2(workgroupSize);
    let treeReductionCode = '';
    for (let step = 0; step < numSteps; step++) {
        const stride = workgroupSize >> (step + 1);
        const accumulateCode = opConfig.accumulator('shared_data[tid]', `shared_data[tid + ${stride}u]`, computeType);
        treeReductionCode += `
        workgroupBarrier();
        if (tid < ${stride}u) {
            ${accumulateCode}
        }
        `;
    }

    const needsF16 = input.dtype === 'float16';
    const enableF16 = needsF16 && resolver.supportsNativeF16 ? 'enable f16;\n' : '';

    return `
${enableF16}
// Global Reduction Stage 1 (CONTIGUOUS): ${dispatchKey}

struct Uniforms {
    numel: u32,
    input_offset: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: ${storageTypeInput};
@group(0) @binding(2) var<storage, read_write> partial_results: array<${computeType}>;

var<workgroup> shared_data: array<${computeType}, ${workgroupSize}>;

${loaderInput.code}

@compute @workgroup_size(${workgroupSize})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let global_id = gid.x;
    
    // Load data to shared memory (or initialize if out of bounds)
    if (global_id < uniforms.numel) {
        let raw_val = ${loaderInput.funcName}(global_id + uniforms.input_offset);
        shared_data[tid] = ${generateCastSnippet('raw_val', elemTypeInput, computeType)};
    } else {
        shared_data[tid] = ${opConfig.initializer(computeType)};
    }
    
    // Tree reduction in shared memory
    ${treeReductionCode}
    
    // Thread 0 writes result
    if (tid == 0u) {
        partial_results[wid.x] = shared_data[0];
    }
}
`;
}

/**
 * Build global reduction Stage 1 shader for STRIDED (non-contiguous) inputs
 * 
 * Key difference from contiguous version:
 * - Uses shape/strides to compute physical offset from logical index
 * - Follows PyTorch ATen design: kernel uses stride info to navigate elements
 */
export function buildStridedGlobalReductionStage1(
    iter: ITensorIterator,
    dispatchKey: string,
    workgroupSize: number,
    rank: number
): string {
    const opConfig = REDUCTION_OPS[dispatchKey];
    if (!opConfig) {
        throw new Error(`Unknown reduction operation: ${dispatchKey}`);
    }

    const input = iter.input(0);
    const resolver = getGlobalDTypeResolver();
    const computeType = getComputeType(iter.computeDtype);
    const storageTypeInput = getStorageType(input.dtype);
    const elemTypeInput = resolver.getDescriptor(input.dtype).wgslStorageType;

    const loaderInput = generateLoadSnippet('input', input.dtype);

    // Generate tree reduction loop
    const numSteps = Math.log2(workgroupSize);
    let treeReductionCode = '';
    for (let step = 0; step < numSteps; step++) {
        const stride = workgroupSize >> (step + 1);
        const accumulateCode = opConfig.accumulator('shared_data[tid]', `shared_data[tid + ${stride}u]`, computeType);
        treeReductionCode += `
        workgroupBarrier();
        if (tid < ${stride}u) {
            ${accumulateCode}
        }
        `;
    }

    const needsF16 = input.dtype === 'float16';
    const enableF16 = needsF16 && resolver.supportsNativeF16 ? 'enable f16;\n' : '';

    return `
${enableF16}
// Global Reduction Stage 1 (STRIDED): ${dispatchKey}, Rank: ${rank}

struct Uniforms {
    numel: u32,
    rank: u32,
    input_offset: u32,
    _pad0: u32,
    // Shape: max 8 dims packed in 2 vec4
    shape0: vec4<u32>,
    shape1: vec4<u32>,
    // Strides: max 8 dims packed in 2 vec4
    strides0: vec4<u32>,
    strides1: vec4<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: ${storageTypeInput};
@group(0) @binding(2) var<storage, read_write> partial_results: array<${computeType}>;

var<workgroup> shared_data: array<${computeType}, ${workgroupSize}>;

${loaderInput.code}

fn get_shape(dim: u32) -> u32 {
    if (dim < 4u) { return uniforms.shape0[dim]; }
    else { return uniforms.shape1[dim - 4u]; }
}

fn get_stride(dim: u32) -> u32 {
    if (dim < 4u) { return uniforms.strides0[dim]; }
    else { return uniforms.strides1[dim - 4u]; }
}

// Compute physical offset from logical flat index using strides
fn compute_strided_offset(flat_idx: u32) -> u32 {
    var offset: u32 = 0u;
    var remaining = flat_idx;
    for (var d: i32 = i32(uniforms.rank) - 1; d >= 0; d = d - 1) {
        let dim = u32(d);
        let dim_size = get_shape(dim);
        let coord = remaining % dim_size;
        remaining = remaining / dim_size;
        offset = offset + coord * get_stride(dim);
    }
    return offset;
}

@compute @workgroup_size(${workgroupSize})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let logical_id = gid.x;
    
    // Load data to shared memory using strided access
    if (logical_id < uniforms.numel) {
        let physical_offset = compute_strided_offset(logical_id);
        let raw_val = ${loaderInput.funcName}(physical_offset + uniforms.input_offset);
        shared_data[tid] = ${generateCastSnippet('raw_val', elemTypeInput, computeType)};
    } else {
        shared_data[tid] = ${opConfig.initializer(computeType)};
    }
    
    // Tree reduction in shared memory
    ${treeReductionCode}
    
    // Thread 0 writes result
    if (tid == 0u) {
        partial_results[wid.x] = shared_data[0];
    }
}
`;
}

/**
 * Build Stage 2 FINAL: Merge partial results and apply finalizer
 */
export function buildGlobalReductionStage2(
    dispatchKey: string,
    computeType: string,
    outputType: string,
    workgroupSize: number,
    totalNumel: number  // For mean normalization
): string {
    const opConfig = REDUCTION_OPS[dispatchKey];
    if (!opConfig) {
        throw new Error(`Unknown reduction operation: ${dispatchKey}`);
    }

    // Generate tree reduction loop
    const numSteps = Math.log2(workgroupSize);
    let treeReductionCode = '';
    for (let step = 0; step < numSteps; step++) {
        const stride = workgroupSize >> (step + 1);
        const accumulateCode = opConfig.accumulator('shared_data[tid]', `shared_data[tid + ${stride}u]`, computeType);
        treeReductionCode += `
        workgroupBarrier();
        if (tid < ${stride}u) {
            ${accumulateCode}
        }
        `;
    }

    const needsFinalizer = opConfig.finalizer !== undefined;

    return `
// Global Reduction Stage 2 FINAL: ${dispatchKey}

struct Uniforms {
    num_partials: u32,
    output_offset: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> partial_results: array<${computeType}>;
@group(0) @binding(2) var<storage, read_write> output: array<${outputType}>;

var<workgroup> shared_data: array<${computeType}, ${workgroupSize}>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    
    // Load partial results into shared memory
    if (tid < uniforms.num_partials) {
        shared_data[tid] = partial_results[tid];
    } else {
        shared_data[tid] = ${opConfig.initializer(computeType)};
    }
    
    // Tree reduction in shared memory
    ${treeReductionCode}
    
    // Thread 0 writes final result
    if (tid == 0u) {
        var result = shared_data[0];
        ${needsFinalizer ? `result = ${opConfig.finalizer!('result', `${totalNumel}u`, computeType)};` : ''}
        output[uniforms.output_offset] = ${computeType === outputType ? 'result' : generateCastSnippet('result', computeType as any, outputType as any)};
    }
}
`;
}

/**
 * Build naive reduction shader for small data (single workgroup) - CONTIGUOUS
 */
export function buildNaiveReductionShader(
    iter: ITensorIterator,
    dispatchKey: string,
    workgroupSize: number
): string {
    const opConfig = REDUCTION_OPS[dispatchKey];
    if (!opConfig) {
        throw new Error(`Unknown reduction operation: ${dispatchKey}`);
    }

    const input = iter.input(0);
    const output = iter.output();
    const resolver = getGlobalDTypeResolver();
    const computeType = getComputeType(iter.computeDtype);
    const storageTypeInput = getStorageType(input.dtype);
    const storageTypeOutput = getStorageType(output.dtype);
    const elemTypeInput = resolver.getDescriptor(input.dtype).wgslStorageType;
    const elemTypeOutput = resolver.getDescriptor(output.dtype).wgslStorageType;

    const loaderInput = generateLoadSnippet('input', input.dtype);

    // Generate tree reduction loop
    const numSteps = Math.log2(workgroupSize);
    let treeReductionCode = '';
    for (let step = 0; step < numSteps; step++) {
        const stride = workgroupSize >> (step + 1);
        const accumulateCode = opConfig.accumulator('shared_data[tid]', `shared_data[tid + ${stride}u]`, computeType);
        treeReductionCode += `
        workgroupBarrier();
        if (tid < ${stride}u) {
            ${accumulateCode}
        }
        `;
    }

    const needsFinalizer = opConfig.finalizer !== undefined;
    const needsF16 = input.dtype === 'float16';
    const enableF16 = needsF16 && resolver.supportsNativeF16 ? 'enable f16;\n' : '';

    return `
${enableF16}
// Naive Reduction (CONTIGUOUS): ${dispatchKey}

struct Uniforms {
    numel: u32,
    input_offset: u32,
    output_offset: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: ${storageTypeInput};
@group(0) @binding(2) var<storage, read_write> output: ${storageTypeOutput};

var<workgroup> shared_data: array<${computeType}, ${workgroupSize}>;

${loaderInput.code}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    
    // Each thread accumulates a portion of the data
    var acc = ${opConfig.initializer(computeType)};
    for (var i = tid; i < uniforms.numel; i = i + ${workgroupSize}u) {
        let raw_val = ${loaderInput.funcName}(i + uniforms.input_offset);
        let val = ${generateCastSnippet('raw_val', elemTypeInput, computeType)};
        ${opConfig.accumulator('acc', 'val', computeType)};
    }
    
    shared_data[tid] = acc;
    
    // Tree reduction in shared memory
    ${treeReductionCode}
    
    // Thread 0 writes final result
    if (tid == 0u) {
        var result = shared_data[0];
        ${needsFinalizer ? `result = ${opConfig.finalizer!('result', 'uniforms.numel', computeType)};` : ''}
        output[uniforms.output_offset] = ${computeType === elemTypeOutput ? 'result' : generateCastSnippet('result', computeType as any, elemTypeOutput)};
    }
}
`;
}

/**
 * Build naive reduction shader for small data (single workgroup) - STRIDED
 * 
 * Uses shape/strides to compute physical offset from logical index
 */
export function buildStridedNaiveReductionShader(
    iter: ITensorIterator,
    dispatchKey: string,
    workgroupSize: number,
    rank: number
): string {
    const opConfig = REDUCTION_OPS[dispatchKey];
    if (!opConfig) {
        throw new Error(`Unknown reduction operation: ${dispatchKey}`);
    }

    const input = iter.input(0);
    const output = iter.output();
    const resolver = getGlobalDTypeResolver();
    const computeType = getComputeType(iter.computeDtype);
    const storageTypeInput = getStorageType(input.dtype);
    const storageTypeOutput = getStorageType(output.dtype);
    const elemTypeInput = resolver.getDescriptor(input.dtype).wgslStorageType;
    const elemTypeOutput = resolver.getDescriptor(output.dtype).wgslStorageType;

    const loaderInput = generateLoadSnippet('input', input.dtype);

    // Generate tree reduction loop
    const numSteps = Math.log2(workgroupSize);
    let treeReductionCode = '';
    for (let step = 0; step < numSteps; step++) {
        const stride = workgroupSize >> (step + 1);
        const accumulateCode = opConfig.accumulator('shared_data[tid]', `shared_data[tid + ${stride}u]`, computeType);
        treeReductionCode += `
        workgroupBarrier();
        if (tid < ${stride}u) {
            ${accumulateCode}
        }
        `;
    }

    const needsFinalizer = opConfig.finalizer !== undefined;
    const needsF16 = input.dtype === 'float16';
    const enableF16 = needsF16 && resolver.supportsNativeF16 ? 'enable f16;\n' : '';

    return `
${enableF16}
// Naive Reduction (STRIDED): ${dispatchKey}, Rank: ${rank}

struct Uniforms {
    numel: u32,
    rank: u32,
    input_offset: u32,
    output_offset: u32,
    // Shape: max 8 dims packed in 2 vec4
    shape0: vec4<u32>,
    shape1: vec4<u32>,
    // Strides: max 8 dims packed in 2 vec4
    strides0: vec4<u32>,
    strides1: vec4<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: ${storageTypeInput};
@group(0) @binding(2) var<storage, read_write> output: ${storageTypeOutput};

var<workgroup> shared_data: array<${computeType}, ${workgroupSize}>;

${loaderInput.code}

fn get_shape(dim: u32) -> u32 {
    if (dim < 4u) { return uniforms.shape0[dim]; }
    else { return uniforms.shape1[dim - 4u]; }
}

fn get_stride(dim: u32) -> u32 {
    if (dim < 4u) { return uniforms.strides0[dim]; }
    else { return uniforms.strides1[dim - 4u]; }
}

// Compute physical offset from logical flat index using strides
fn compute_strided_offset(flat_idx: u32) -> u32 {
    var offset: u32 = 0u;
    var remaining = flat_idx;
    for (var d: i32 = i32(uniforms.rank) - 1; d >= 0; d = d - 1) {
        let dim = u32(d);
        let dim_size = get_shape(dim);
        let coord = remaining % dim_size;
        remaining = remaining / dim_size;
        offset = offset + coord * get_stride(dim);
    }
    return offset;
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    
    // Each thread accumulates a portion of the data using strided access
    var acc = ${opConfig.initializer(computeType)};
    for (var i = tid; i < uniforms.numel; i = i + ${workgroupSize}u) {
        let physical_offset = compute_strided_offset(i);
        let raw_val = ${loaderInput.funcName}(physical_offset + uniforms.input_offset);
        let val = ${generateCastSnippet('raw_val', elemTypeInput, computeType)};
        ${opConfig.accumulator('acc', 'val', computeType)};
    }
    
    shared_data[tid] = acc;
    
    // Tree reduction in shared memory
    ${treeReductionCode}
    
    // Thread 0 writes final result
    if (tid == 0u) {
        var result = shared_data[0];
        ${needsFinalizer ? `result = ${opConfig.finalizer!('result', 'uniforms.numel', computeType)};` : ''}
        output[uniforms.output_offset] = ${computeType === elemTypeOutput ? 'result' : generateCastSnippet('result', computeType as any, elemTypeOutput)};
    }
}
`;
}

