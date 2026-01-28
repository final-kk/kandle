/**
 * Flip Shader Builder
 *
 * Generates WGSL shader for flip operation (reversing elements along specified dimensions).
 */

import { getGlobalDTypeResolver } from '../../../base/DTypeResolver';
import type { DType } from '@kandle/types';

const MAX_RANK = 8;

export interface FlipShaderConfig {
    dtype: DType;
    rank: number;
}

/**
 * Build WGSL shader for flip kernel
 */
export function buildFlipShader(config: FlipShaderConfig): string {
    const resolver = getGlobalDTypeResolver();
    const wgslType = resolver.getDescriptor(config.dtype).wgslStorageType;
    const rank = config.rank;

    return /* wgsl */ `
// Flip Kernel Shader
// Reverses elements along specified dimensions

struct Uniforms {
    numel: u32,
    rank: u32,
    input_offset: u32,
    output_offset: u32,
    flip_mask: u32,  // Bitmask of dimensions to flip
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    input_shape: array<u32, ${MAX_RANK}>,
    input_strides: array<i32, ${MAX_RANK}>,
    output_strides: array<i32, ${MAX_RANK}>,
}

@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<${wgslType}>;
@group(0) @binding(2) var<storage, read_write> output: array<${wgslType}>;

// Convert flat output index to N-D coordinates
fn idx_to_coord(idx: u32) -> array<u32, ${MAX_RANK}> {
    var coords: array<u32, ${MAX_RANK}>;
    var remaining = idx;
    
    for (var d = 0u; d < ${rank}u; d++) {
        let stride = u32(uniforms.output_strides[d]);
        if (stride > 0u) {
            coords[d] = remaining / stride;
            remaining = remaining % stride;
        } else {
            coords[d] = 0u;
        }
    }
    
    return coords;
}

// Compute input linear index from potentially flipped coordinates
fn compute_input_idx(out_coords: array<u32, ${MAX_RANK}>) -> u32 {
    var idx = uniforms.input_offset;
    let flip_mask = uniforms.flip_mask;
    
    for (var d = 0u; d < ${rank}u; d++) {
        var coord = out_coords[d];
        
        // Check if this dimension should be flipped
        if ((flip_mask & (1u << d)) != 0u) {
            // Flip: idx = shape[d] - 1 - idx
            coord = uniforms.input_shape[d] - 1u - coord;
        }
        
        idx += coord * u32(uniforms.input_strides[d]);
    }
    
    return idx;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_idx = gid.x;
    if (out_idx >= uniforms.numel) {
        return;
    }
    
    // Convert output index to N-D coordinates
    let out_coords = idx_to_coord(out_idx);
    
    // Compute input index with flipping applied
    let in_idx = compute_input_idx(out_coords);
    
    // Copy with flip
    output[uniforms.output_offset + out_idx] = input[in_idx];
}
`;
}
