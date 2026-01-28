/**
 * Copy Kernel Shader Builder
 * 
 * Generates WGSL shaders for copy operations:
 * - cast: type conversion between dtypes
 * - contiguous: strided to contiguous conversion
 * - clone: direct memory copy
 * 
 * v5 FIX: Shape/Strides are passed via Uniforms (not hardcoded constants)
 * Uses correct WGSL memory alignment with vec4 types
 */

import { DType } from '@kandle/types';
import { getStorageType } from '../../shader/ShaderSnippets';
import { getGlobalDTypeResolver } from '../../base/DTypeResolver';
import { isContiguousStrides } from '@kandle/utils';
import type { CopyOpConfig } from './types';

const MAX_RANK = 8;

/**
 * Build copy shader based on configuration
 */
export function buildCopyShader(config: CopyOpConfig): string {
    const {
        inputDtype,
        outputDtype,
        shape,
        inputStrides,
        outputStrides,
        inputOffset,
        outputOffset,
        numel
    } = config;

    const rank = shape.length;

    // Get storage types
    const storageTypeIn = getStorageType(inputDtype);
    const storageTypeOut = getStorageType(outputDtype);

    // Get element types
    const resolver = getGlobalDTypeResolver();
    const elemTypeIn = resolver.getDescriptor(inputDtype).wgslStorageType;
    const elemTypeOut = resolver.getDescriptor(outputDtype).wgslStorageType;

    // Check if type conversion is needed
    const needsCast = inputDtype !== outputDtype;

    // Check contiguity (fast path detection)
    const isInputContiguous = isContiguousStrides(shape, inputStrides);
    const isOutputContiguous = isContiguousStrides(shape, outputStrides);

    // Fast path: contiguous input/output with zero offsets
    if (isInputContiguous && isOutputContiguous && inputOffset === 0 && outputOffset === 0) {
        return generateFastCopyShader(
            storageTypeIn,
            storageTypeOut,
            elemTypeIn,
            elemTypeOut,
            needsCast
        );
    }

    // Slow path: strided copy with Uniforms (cacheable pipeline)
    return generateStridedCopyShader(
        storageTypeIn,
        storageTypeOut,
        elemTypeIn,
        elemTypeOut,
        needsCast,
        rank
    );
}

/**
 * Fast path: contiguous copy (possibly with type conversion)
 * Uses uniforms for numel and offsets only
 */
function generateFastCopyShader(
    storageTypeIn: string,
    storageTypeOut: string,
    elemTypeIn: string,
    elemTypeOut: string,
    needsCast: boolean
): string {
    const castExpr = needsCast ? `${elemTypeOut}(val)` : 'val';

    return `// Fast Copy Shader (contiguous) - Pipeline Cacheable
struct Uniforms {
    numel: u32,
    offset_input: u32,
    offset_output: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: ${storageTypeIn};
@group(0) @binding(2) var<storage, read_write> output: ${storageTypeOut};

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.numel) {
        return;
    }

    let val = input[idx + uniforms.offset_input];
    output[idx + uniforms.offset_output] = ${castExpr};
}
`;
}

/**
 * Slow path: strided copy
 * Uses correct WGSL memory alignment with vec4 types
 */
function generateStridedCopyShader(
    storageTypeIn: string,
    storageTypeOut: string,
    elemTypeIn: string,
    elemTypeOut: string,
    needsCast: boolean,
    rank: number
): string {
    const castExpr = needsCast ? `${elemTypeOut}(val)` : 'val';

    return `// Strided Copy Shader - Pipeline Cacheable (Rank ${rank})
struct Uniforms {
    numel: u32,
    rank: u32,
    offset_input: u32,
    offset_output: u32,
    shape0: vec4<u32>,       // shape[0..3]
    shape1: vec4<u32>,       // shape[4..7]
    input_strides0: vec4<i32>,   // input_strides[0..3]
    input_strides1: vec4<i32>,   // input_strides[4..7]
    output_strides0: vec4<i32>,  // output_strides[0..3]
    output_strides1: vec4<i32>,  // output_strides[4..7]
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: ${storageTypeIn};
@group(0) @binding(2) var<storage, read_write> output: ${storageTypeOut};

fn get_shape(dim: u32) -> u32 {
    if (dim < 4u) {
        return uniforms.shape0[dim];
    } else {
        return uniforms.shape1[dim - 4u];
    }
}

fn get_input_stride(dim: u32) -> i32 {
    if (dim < 4u) {
        return uniforms.input_strides0[dim];
    } else {
        return uniforms.input_strides1[dim - 4u];
    }
}

fn get_output_stride(dim: u32) -> i32 {
    if (dim < 4u) {
        return uniforms.output_strides0[dim];
    } else {
        return uniforms.output_strides1[dim - 4u];
    }
}

// Convert flat index to multi-dimensional indices
fn unravel_index(flat_idx: u32) -> array<u32, ${MAX_RANK}> {
    var indices: array<u32, ${MAX_RANK}>;
    var remaining = flat_idx;
    
    for (var i: i32 = i32(uniforms.rank) - 1; i >= 0; i = i - 1) {
        let dim = u32(i);
        indices[dim] = remaining % get_shape(dim);
        remaining = remaining / get_shape(dim);
    }
    
    return indices;
}

// Compute strided input index
fn compute_input_index(indices: array<u32, ${MAX_RANK}>) -> i32 {
    var idx: i32 = i32(uniforms.offset_input);
    for (var i: u32 = 0u; i < uniforms.rank; i = i + 1u) {
        idx = idx + i32(indices[i]) * get_input_stride(i);
    }
    return idx;
}

// Compute strided output index
fn compute_output_index(indices: array<u32, ${MAX_RANK}>) -> i32 {
    var idx: i32 = i32(uniforms.offset_output);
    for (var i: u32 = 0u; i < uniforms.rank; i = i + 1u) {
        idx = idx + i32(indices[i]) * get_output_stride(i);
    }
    return idx;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_flat_idx = gid.x;
    if (output_flat_idx >= uniforms.numel) {
        return;
    }

    // Convert flat output index to multi-dimensional indices
    let indices = unravel_index(output_flat_idx);
    
    // Compute strided input index
    let input_idx = compute_input_index(indices);
    
    // Compute strided output index
    let output_idx = compute_output_index(indices);
    
    // Load, optionally cast, and store
    let val = input[input_idx];
    output[output_idx] = ${castExpr};
}
`;
}

/**
 * Get pipeline key for caching
 * Only depends on (rank, inputDtype, outputDtype, isContiguous)
 */
export function getCopyPipelineKey(config: CopyOpConfig): string {
    const { inputDtype, outputDtype, shape, inputStrides, outputStrides, inputOffset, outputOffset } = config;
    const rank = shape.length;

    const isInputContiguous = isContiguousStrides(shape, inputStrides);
    const isOutputContiguous = isContiguousStrides(shape, outputStrides);
    const isFastPath = isInputContiguous && isOutputContiguous && inputOffset === 0 && outputOffset === 0;

    const path = isFastPath ? 'fast' : `strided-r${rank}`;
    return `copy.${inputDtype}.${outputDtype}.${path}`;
}
