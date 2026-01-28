/**
 * RepeatInterleave Shader Builder
 * 
 * Generates WGSL shaders for repeat_interleave operation
 * 
 * Algorithm:
 * For scalar repeats along dim:
 * - output[..., i*repeats+r, ...] = input[..., i, ...]
 * 
 * For flattened (dim=None):
 * - output[i*repeats+r] = input[i]
 */

import { DType } from '@kandle/types';
import { getGlobalDTypeResolver } from '../../base/DTypeResolver';
import type { RepeatInterleaveShaderParams } from './repeatInterleaveTypes';

function getWgslType(dtype: DType): string {
    const resolver = getGlobalDTypeResolver();
    return resolver.getDescriptor(dtype).wgslComputeType;
}

/**
 * Build the complete shader for repeat_interleave
 */
export function buildRepeatInterleaveShader(params: RepeatInterleaveShaderParams): string {
    const { dtype, rank, hasDim } = params;
    const wgslType = getWgslType(dtype);

    const parts: string[] = [];

    // 1. Uniforms struct
    parts.push(/* wgsl */`
struct Uniforms {
    numel: u32,
    repeats: u32,
    input_numel: u32,
    rank: u32,
    dim: i32,
    input_offset: u32,
    output_offset: u32,
    _pad: u32,
    input_shape: vec4<u32>,
    input_strides: vec4<i32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<${wgslType}>;
@group(0) @binding(2) var<storage, read_write> output: array<${wgslType}>;
`);

    // 2. Main function
    if (hasDim) {
        parts.push(generateDimRepeatMain(rank, wgslType));
    } else {
        parts.push(generateFlatRepeatMain(wgslType));
    }

    return parts.join('\n');
}

/**
 * Generate main function for flat repeat (dim=None)
 * 
 * When dim is not specified, input is flattened first, then each element is repeated.
 * output[i * repeats + r] = input[i] for r in 0..repeats
 */
function generateFlatRepeatMain(wgslType: string): string {
    return /* wgsl */`
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_idx = gid.x;
    
    if (output_idx >= uniforms.numel) {
        return;
    }
    
    // For flat repeat: output_idx = input_idx * repeats + repeat_idx
    let input_idx = output_idx / uniforms.repeats;
    
    // Read from input (treating as flat, but respecting strides)
    // Convert flat input_idx to multi-dimensional coords, then to physical address
    var remaining = input_idx;
    var src_offset: i32 = i32(uniforms.input_offset);
    
    // Unroll for up to 4D
    if (uniforms.rank > 0u) {
        let dim_size_0 = select(uniforms.input_shape[1] * uniforms.input_shape[2] * uniforms.input_shape[3], 1u, uniforms.rank <= 1u);
        let coord_0 = remaining / dim_size_0;
        remaining = remaining % dim_size_0;
        src_offset += i32(coord_0) * uniforms.input_strides[0];
    }
    if (uniforms.rank > 1u) {
        let dim_size_1 = select(uniforms.input_shape[2] * uniforms.input_shape[3], 1u, uniforms.rank <= 2u);
        let coord_1 = remaining / dim_size_1;
        remaining = remaining % dim_size_1;
        src_offset += i32(coord_1) * uniforms.input_strides[1];
    }
    if (uniforms.rank > 2u) {
        let dim_size_2 = select(uniforms.input_shape[3], 1u, uniforms.rank <= 3u);
        let coord_2 = remaining / dim_size_2;
        remaining = remaining % dim_size_2;
        src_offset += i32(coord_2) * uniforms.input_strides[2];
    }
    if (uniforms.rank > 3u) {
        let coord_3 = remaining;
        src_offset += i32(coord_3) * uniforms.input_strides[3];
    }
    
    output[output_idx + uniforms.output_offset] = input[u32(src_offset)];
}
`;
}

/**
 * Generate main function for repeat along specific dimension
 * 
 * For dim-specific repeat:
 * output[..., output_dim_idx, ...] = input[..., output_dim_idx / repeats, ...]
 */
function generateDimRepeatMain(rank: number, wgslType: string): string {
    return /* wgsl */`
// Helper to get shape element
fn get_shape(d: u32) -> u32 {
    return uniforms.input_shape[d];
}

// Helper to get stride element
fn get_stride(d: u32) -> i32 {
    return uniforms.input_strides[d];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_idx = gid.x;
    
    if (output_idx >= uniforms.numel) {
        return;
    }
    
    // Compute output coordinates from flat index
    // Output shape has input_shape[dim] * repeats at dimension dim
    var remaining = output_idx;
    var src_offset: i32 = i32(uniforms.input_offset);
    
    let dim = u32(max(0, uniforms.dim));
    
    // For each dimension, compute coordinate and contribution to source offset
    for (var d = 0u; d < uniforms.rank; d = d + 1u) {
        // Compute suffix product (size of all dimensions after d)
        var suffix_size = 1u;
        for (var j = d + 1u; j < uniforms.rank; j = j + 1u) {
            if (j == dim) {
                // Output size at dim is input_shape[dim] * repeats
                suffix_size = suffix_size * get_shape(j) * uniforms.repeats;
            } else {
                suffix_size = suffix_size * get_shape(j);
            }
        }
        
        // Compute coordinate at this dimension
        var coord: u32;
        if (d == dim) {
            // At the repeat dimension, output size is input_size * repeats
            let output_dim_size = get_shape(d) * uniforms.repeats;
            coord = remaining / suffix_size;
            remaining = remaining % suffix_size;
            // Map back to input coordinate: input_coord = output_coord / repeats
            let input_coord = coord / uniforms.repeats;
            src_offset += i32(input_coord) * get_stride(d);
        } else {
            coord = remaining / suffix_size;
            remaining = remaining % suffix_size;
            src_offset += i32(coord) * get_stride(d);
        }
    }
    
    output[output_idx + uniforms.output_offset] = input[u32(src_offset)];
}
`;
}
