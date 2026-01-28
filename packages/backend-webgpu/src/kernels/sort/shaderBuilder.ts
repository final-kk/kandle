/**
 * Sort Shader Builder (v5)
 *
 * Generates WGSL compute shaders for sort operations using bitonic sort algorithm.
 * 
 * Bitonic Sort Overview:
 * - O(n log²n) parallel sorting algorithm
 * - GPU-friendly: highly parallelizable, regular memory access patterns
 * - Works by building progressively larger "bitonic sequences" and merging them
 * 
 * Implementation Notes:
 * - Each workgroup sorts one independent "slice" of the input tensor
 * - Uses shared memory for efficiency within workgroup
 * - Supports strided tensor access via shape/strides uniforms
 * - Supports multi-dtype via DTypeResolver
 * 
 * References:
 * - Batcher, K. E. (1968). "Sorting networks and their applications"
 * - GPU Gems 2, Chapter 46: "Improved GPU Sorting"
 * - https://en.wikipedia.org/wiki/Bitonic_sorter
 */

import type { DType } from '@kandle/types';
import type { PhysicalStorageDescriptor } from '../../base/DTypeResolver';
import { getGlobalDTypeResolver } from '../../base/DTypeResolver';
import { getStorageType, generateLoadSnippet } from '../../shader/ShaderSnippets';
import { generateCastSnippet } from '../../base/dtype';
import type { SortConfig, SortScalarArgs } from './types';

const MAX_RANK = 8;

/**
 * Build WGSL shader for sort operation
 * 
 * Dispatches to appropriate algorithm implementation
 */
export function buildSortShader(
    config: SortConfig,
    descriptor: PhysicalStorageDescriptor
): string {
    const { opConfig } = config;

    switch (opConfig.algorithm) {
        case 'bitonic':
            return buildBitonicSortShader(config, descriptor);
        case 'radix_select':
            throw new Error('Radix select not yet implemented');
        case 'radix_sort':
            throw new Error('Radix sort not yet implemented');
        default:
            throw new Error(`Unknown sort algorithm: ${opConfig.algorithm}`);
    }
}

/**
 * Build uniform struct declaration for sort operations
 * 
 * Layout (aligned to 16-byte boundaries):
 * - Header: dim_size, num_slices, sort_dim, output_dim_size = 16 bytes
 * - Input shape: vec4 * 2 = 32 bytes
 * - Input strides: vec4 * 2 = 32 bytes
 * - Output strides: vec4 * 2 = 32 bytes
 * - Offsets + scalars: aligned to 16 bytes
 */
function buildUniformsStruct(scalars: SortScalarArgs): string {
    // Build scalar fields based on what's present
    const scalarFields: string[] = [];

    // descending/largest are effectively the same flag (just inverted semantics)
    scalarFields.push('descending: u32,');  // 1 = descending/largest, 0 = ascending/smallest
    scalarFields.push('sorted_flag: u32,');  // For topk: whether to maintain sorted order
    scalarFields.push('_pad_scalars0: u32,');
    scalarFields.push('_pad_scalars1: u32,');

    return `
struct Uniforms {
    // Header (16 bytes)
    dim_size: u32,         // Size of dimension being sorted
    num_slices: u32,       // Number of independent slices
    sort_dim: u32,         // Which dimension to sort along
    output_dim_size: u32,  // Output dimension size (k for topk, dim_size for sort)
    
    // Input shape (32 bytes)
    input_shape0: vec4<u32>,
    input_shape1: vec4<u32>,
    
    // Input strides (32 bytes)
    input_strides0: vec4<u32>,
    input_strides1: vec4<u32>,
    
    // Output strides (32 bytes)
    output_strides0: vec4<u32>,
    output_strides1: vec4<u32>,
    
    // Offsets (16 bytes)
    input_offset: u32,
    values_offset: u32,
    indices_offset: u32,
    rank: u32,
    
    // Scalar arguments (16 bytes)
    ${scalarFields.join('\n    ')}
}
`;
}

/**
 * Build helper functions for strided tensor access
 */
function buildStridedAccessHelpers(): string {
    return `
// Get shape value for a dimension
fn get_input_shape(dim: u32) -> u32 {
    if (dim < 4u) { return uniforms.input_shape0[dim]; }
    else { return uniforms.input_shape1[dim - 4u]; }
}

// Get input stride for a dimension
fn get_input_stride(dim: u32) -> u32 {
    if (dim < 4u) { return uniforms.input_strides0[dim]; }
    else { return uniforms.input_strides1[dim - 4u]; }
}

// Get output stride for a dimension
fn get_output_stride(dim: u32) -> u32 {
    if (dim < 4u) { return uniforms.output_strides0[dim]; }
    else { return uniforms.output_strides1[dim - 4u]; }
}

// Calculate input index from (slice_idx, dim_coord)
// slice_idx: which independent slice we're sorting
// dim_coord: position within the sorted dimension
fn calculate_input_index(slice_idx: u32, dim_coord: u32) -> u32 {
    var linear_idx = uniforms.input_offset;
    var remaining = slice_idx;
    
    // Decompose slice_idx into multi-dimensional coordinates
    // Process dimensions in reverse order (row-major), skipping sort dimension
    for (var d_iter: i32 = i32(uniforms.rank) - 1; d_iter >= 0; d_iter = d_iter - 1) {
        let d = u32(d_iter);
        if (d == uniforms.sort_dim) {
            continue;  // Skip sort dimension
        }
        
        let dim_size = get_input_shape(d);
        if (dim_size > 1u) {
            let coord = remaining % dim_size;
            linear_idx = linear_idx + coord * get_input_stride(d);
            remaining = remaining / dim_size;
        }
    }
    
    // Add sort dimension coordinate
    linear_idx = linear_idx + dim_coord * get_input_stride(uniforms.sort_dim);
    
    return linear_idx;
}

// Calculate output index from (slice_idx, out_coord)
fn calculate_output_index(slice_idx: u32, out_coord: u32) -> u32 {
    var linear_idx: u32 = 0u;
    var remaining = slice_idx;
    
    // Decompose slice_idx into multi-dimensional coordinates
    for (var d_iter: i32 = i32(uniforms.rank) - 1; d_iter >= 0; d_iter = d_iter - 1) {
        let d = u32(d_iter);
        if (d == uniforms.sort_dim) {
            continue;  // Skip sort dimension
        }
        
        let dim_size = get_input_shape(d);
        if (dim_size > 1u) {
            let coord = remaining % dim_size;
            linear_idx = linear_idx + coord * get_output_stride(d);
            remaining = remaining / dim_size;
        }
    }
    
    // Add sort dimension coordinate
    linear_idx = linear_idx + out_coord * get_output_stride(uniforms.sort_dim);
    
    return linear_idx;
}
`;
}

/**
 * Build bitonic sort shader
 * 
 * Algorithm:
 * 1. Load data into shared memory
 * 2. Perform bitonic sort iterations
 * 3. Extract top-k or all elements
 * 4. Write to output buffers
 */
function buildBitonicSortShader(
    config: SortConfig,
    descriptor: PhysicalStorageDescriptor
): string {
    const { dimSize, outputDimSize, opConfig, dtype, scalars } = config;

    const resolver = getGlobalDTypeResolver();
    const storageType = getStorageType(dtype);
    const elemType = descriptor.wgslStorageType;
    const computeType = descriptor.wgslComputeType;

    const loaderInput = generateLoadSnippet('input', dtype);

    // Determine workgroup size: must be power of 2, at least dimSize
    // Cap at 256 for most GPUs (can be tuned)
    const maxWorkgroupSize = 256;
    const workgroupSize = Math.min(
        maxWorkgroupSize,
        nextPowerOf2(dimSize)
    );

    // For large arrays, each thread may need to process multiple elements
    const elementsPerThread = Math.ceil(dimSize / workgroupSize);

    // Pad to power of 2 for bitonic sort
    const paddedSize = nextPowerOf2(dimSize);

    // Number of stages (must be integer for shader)
    const numStages = Math.ceil(Math.log2(paddedSize));
    // Use integer division to avoid "0.5u" in shader when paddedSize=1
    const halfPaddedSize = Math.floor(paddedSize / 2);

    const needsF16 = dtype === 'float16';
    const enableF16 = needsF16 && resolver.supportsNativeF16 ? 'enable f16;\n' : '';

    // Generate comparison expression based on dtype
    const compareExpr = generateCompareExpr(computeType);

    // Build output bindings based on what's needed
    let bindingIdx = 2;  // 0=uniforms, 1=input
    const outputBindings: string[] = [];
    const outputWrites: string[] = [];

    if (opConfig.needsValues) {
        outputBindings.push(
            `@group(0) @binding(${bindingIdx++}) var<storage, read_write> output_values: ${storageType};`
        );
    }
    if (opConfig.needsIndices) {
        outputBindings.push(
            `@group(0) @binding(${bindingIdx}) var<storage, read_write> output_indices: array<i32>;`
        );
    }

    // Generate the main shader
    return `
${enableF16}
// ============================================================
// Bitonic Sort Shader - ${config.dispatchKey}
// ============================================================
// Configuration:
//   - Input dim size: ${dimSize}
//   - Padded size: ${paddedSize} (next power of 2)
//   - Workgroup size: ${workgroupSize}
//   - Output dim size: ${outputDimSize}
//   - Dtype: ${dtype}
//   - Storage type: ${storageType}
//   - Compute type: ${computeType}
//   - Needs values: ${opConfig.needsValues}
//   - Needs indices: ${opConfig.needsIndices}
// 
// Uniform descending flag interpretation:
//   - descending=0 → ascending sort (smallest first)
//   - descending=1 → descending sort (largest first)
// ============================================================

${buildUniformsStruct(scalars)}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: ${storageType};
${outputBindings.join('\n')}

// Shared memory for values and indices
var<workgroup> shared_values: array<${computeType}, ${paddedSize}>;
var<workgroup> shared_indices: array<i32, ${paddedSize}>;

${loaderInput.code}

${buildStridedAccessHelpers()}

// Comparison function
// Returns true if (a, idx_a) should come before (b, idx_b)
fn should_swap(a: ${computeType}, b: ${computeType}, idx_a: i32, idx_b: i32, ascending: bool) -> bool {
    ${compareExpr}
}

@compute @workgroup_size(${workgroupSize})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let slice_idx = wid.x;
    
    if (slice_idx >= uniforms.num_slices) {
        return;
    }
    
    let ascending = uniforms.descending == 0u;
    let dim_size = uniforms.dim_size;
    let padded_size = ${paddedSize}u;
    
    // ========================================
    // Phase 1: Load data into shared memory
    // ========================================
    
    // Each thread loads one or more elements
    for (var elem: u32 = tid; elem < padded_size; elem = elem + ${workgroupSize}u) {
        if (elem < dim_size) {
            let input_idx = calculate_input_index(slice_idx, elem);
            let raw_val = ${loaderInput.funcName}(input_idx);
            shared_values[elem] = ${generateCastSnippet('raw_val', elemType, computeType)};
            shared_indices[elem] = i32(elem);
        } else {
            // Padding: use sentinel value
            ${generatePaddingSentinel(computeType, 'shared_values[elem]', 'ascending')}
            shared_indices[elem] = i32(elem);
        }
    }
    
    workgroupBarrier();
    
    // ========================================
    // Phase 2: Bitonic Sort
    // ========================================
    
    // Bitonic sort stages
    for (var stage: u32 = 1u; stage <= ${numStages}u; stage = stage + 1u) {
        let stage_size = 1u << stage;  // 2^stage
        
        for (var substage: u32 = stage; substage >= 1u; substage = substage - 1u) {
            let substage_size = 1u << substage;  // 2^substage
            let half_substage = substage_size >> 1u;
            
            // Each thread handles one comparison
            for (var idx: u32 = tid; idx < ${halfPaddedSize}u; idx = idx + ${workgroupSize}u) {
                // Calculate which pair this thread compares
                let group_idx = idx / half_substage;
                let local_idx = idx % half_substage;
                
                let i = group_idx * substage_size + local_idx;
                let j = i + half_substage;
                
                // Determine sort direction based on position in the bitonic sequence
                let block = i / stage_size;
                let sort_ascending = ((block & 1u) == 0u) == ascending;
                
                // Compare and swap if needed
                let val_i = shared_values[i];
                let val_j = shared_values[j];
                let idx_i = shared_indices[i];
                let idx_j = shared_indices[j];
                
                let need_swap = should_swap(val_i, val_j, idx_i, idx_j, sort_ascending);
                
                if (need_swap) {
                    shared_values[i] = val_j;
                    shared_values[j] = val_i;
                    shared_indices[i] = idx_j;
                    shared_indices[j] = idx_i;
                }
            }
            
            workgroupBarrier();
        }
    }
    
    // ========================================
    // Phase 3: Write output
    // ========================================
    
    let output_size = uniforms.output_dim_size;
    
    for (var out_idx: u32 = tid; out_idx < output_size; out_idx = out_idx + ${workgroupSize}u) {
        let out_linear = calculate_output_index(slice_idx, out_idx);
        
${opConfig.needsValues ? `
        // Write values
        let val = shared_values[out_idx];
        output_values[uniforms.values_offset + out_linear] = ${computeType === elemType ? 'val' : generateCastSnippet('val', computeType, elemType)};
` : ''}
${opConfig.needsIndices ? `
        // Write indices
        output_indices[uniforms.indices_offset + out_linear] = shared_indices[out_idx];
` : ''}
    }
}
`;
}

/**
 * Generate comparison expression based on compute type
 */
function generateCompareExpr(computeType: string): string {
    // For floating point, handle NaN: NaNs should sort to end
    if (computeType === 'f32' || computeType === 'f16') {
        return `
    // Handle NaN: NaNs should sort to the end (largest)
    let a_is_nan = a != a;
    let b_is_nan = b != b;
    
    if (a_is_nan && b_is_nan) {
        // Both NaN: use index for stability
        return !ascending && idx_a > idx_b;
    }
    if (a_is_nan) {
        // a is NaN, b is not: a should go to end
        return !ascending;
    }
    if (b_is_nan) {
        // b is NaN, a is not: b should go to end
        return ascending;
    }
    
    // Normal comparison
    if (ascending) {
        if (a == b) {
            // Stable sort: use original index
            return idx_a > idx_b;
        }
        return a > b;
    } else {
        if (a == b) {
            return idx_a > idx_b;
        }
        return a < b;
    }
`;
    }

    // For integer types, simpler comparison (no NaN)
    return `
    if (ascending) {
        if (a == b) {
            return idx_a > idx_b;
        }
        return a > b;
    } else {
        if (a == b) {
            return idx_a > idx_b;
        }
        return a < b;
    }
`;
}

/**
 * Generate padding sentinel value
 * 
 * For ascending sort: padding should be maximum value (sorts to end)
 * For descending sort: padding should be minimum value (sorts to end)
 */
function generatePaddingSentinel(
    computeType: string,
    targetVar: string,
    ascendingVar: string
): string {
    if (computeType === 'f32') {
        return `
            if (${ascendingVar}) {
                ${targetVar} = 3.402823e+38;  // f32 max (sorts to end in ascending)
            } else {
                ${targetVar} = -3.402823e+38;  // f32 min (sorts to end in descending)
            }
`;
    }
    if (computeType === 'f16') {
        return `
            if (${ascendingVar}) {
                ${targetVar} = f16(65504.0);  // f16 max
            } else {
                ${targetVar} = f16(-65504.0);  // f16 min
            }
`;
    }
    if (computeType === 'i32') {
        return `
            if (${ascendingVar}) {
                ${targetVar} = 2147483647;  // i32 max
            } else {
                ${targetVar} = -2147483648;  // i32 min
            }
`;
    }
    if (computeType === 'u32') {
        return `
            if (${ascendingVar}) {
                ${targetVar} = 4294967295u;  // u32 max
            } else {
                ${targetVar} = 0u;  // u32 min
            }
`;
    }

    // Default for unknown types
    return `${targetVar} = ${computeType}(0);`;
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

/**
 * Calculate number of bitonic sort stages
 */
export function calculateBitonicStages(n: number): number {
    return Math.ceil(Math.log2(n));
}
