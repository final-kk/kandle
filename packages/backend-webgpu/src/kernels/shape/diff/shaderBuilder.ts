/**
 * Diff Kernel Shader Builder
 * 
 * Generates WGSL shader for N-order forward difference.
 * 
 * diff(input, n=1, dim=-1):
 *   out[i] = input[i+1] - input[i]  (1st order)
 *   Higher orders: apply 1st order recursively
 */

import type { DType } from '@kandle/types';
import { getGlobalDTypeResolver } from '../../../base/DTypeResolver';

const MAX_RANK = 8;

export interface DiffShaderParams {
    dtype: DType;
    rank: number;
}

/**
 * Build diff kernel shader
 */
export function buildDiffShader(params: DiffShaderParams): string {
    const { dtype, rank } = params;
    const resolver = getGlobalDTypeResolver();
    const storageType = resolver.getDescriptor(dtype).wgslStorageType;
    const workgroupSize = 256;

    return `
// Diff Kernel - N-order forward difference
// dtype: ${dtype}, rank: ${rank}

struct Uniforms {
    numel: u32,
    n: u32,
    dim: i32,
    rank: u32,
    input_offset: u32,
    output_offset: u32,
    _pad0: u32,
    _pad1: u32,
    input_shape: array<u32, ${MAX_RANK}>,
    input_strides: array<i32, ${MAX_RANK}>,
    output_shape: array<u32, ${MAX_RANK}>,
    output_strides: array<i32, ${MAX_RANK}>,
};

@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<${storageType}>;
@group(0) @binding(2) var<storage, read_write> output: array<${storageType}>;

// Convert linear output index to multi-dimensional coordinates
fn idx_to_coords(idx: u32, shape: array<u32, ${MAX_RANK}>, rank: u32) -> array<u32, ${MAX_RANK}> {
    var coords: array<u32, ${MAX_RANK}>;
    var remaining = idx;
    for (var d = i32(rank) - 1; d >= 0; d = d - 1) {
        let dim_size = shape[d];
        coords[d] = remaining % dim_size;
        remaining = remaining / dim_size;
    }
    return coords;
}

// Convert multi-dimensional coordinates to linear index using strides
fn coords_to_offset(coords: array<u32, ${MAX_RANK}>, strides: array<i32, ${MAX_RANK}>, rank: u32) -> u32 {
    var offset: i32 = 0;
    for (var d = 0u; d < rank; d = d + 1u) {
        offset = offset + i32(coords[d]) * strides[d];
    }
    return u32(offset);
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.numel) { return; }

    // Get output coordinates
    let out_coords = idx_to_coords(idx, uniforms.output_shape, uniforms.rank);
    
    // Build coordinates for input[i] and input[i+1] along dim
    var coords_curr = out_coords;
    var coords_next = out_coords;
    
    // Adjust for the difference: output[i] corresponds to input[i+1] - input[i]
    // For 1st order diff: out[i] = in[i+1] - in[i]
    // The output dimension is smaller by n elements
    let dim = u32(uniforms.dim);
    
    // First order difference
    let curr_offset = coords_to_offset(coords_curr, uniforms.input_strides, uniforms.rank);
    coords_next[dim] = coords_next[dim] + 1u;
    let next_offset = coords_to_offset(coords_next, uniforms.input_strides, uniforms.rank);
    
    let val_curr = input[curr_offset + uniforms.input_offset];
    let val_next = input[next_offset + uniforms.input_offset];
    
    let result = val_next - val_curr;
    
    let out_offset = coords_to_offset(out_coords, uniforms.output_strides, uniforms.rank);
    output[out_offset + uniforms.output_offset] = result;
}
`;
}
