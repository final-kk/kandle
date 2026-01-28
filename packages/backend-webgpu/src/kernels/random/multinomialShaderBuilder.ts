/**
 * Multinomial Shader Builder
 * 
 * Generates WGSL shaders for multinomial sampling
 * 
 * Algorithm:
 * 1. Normalize input probabilities (sum to 1)
 * 2. Compute CDF (cumulative distribution function)
 * 3. Generate uniform random numbers
 * 4. Binary search in CDF to find sample indices
 */

import { PHILOX_WGSL_CORE } from './philox.wgsl';
import type { MultinomialShaderParams } from './multinomialTypes';

/**
 * Multinomial uniforms WGSL declaration
 */
const MULTINOMIAL_UNIFORMS_WGSL = /* wgsl */`
struct MultinomialUniforms {
    // Basic info (16 bytes)
    batch_size: u32,
    num_classes: u32,
    num_samples: u32,
    replacement: u32,
    
    // Offsets and padding (16 bytes)
    input_offset: u32,
    output_offset: u32,
    _pad0: u32,
    _pad1: u32,
    
    // Philox Key (16 bytes)
    key0: u32,
    key1: u32,
    base_offset: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> uniforms: MultinomialUniforms;
`;

/**
 * Build the complete shader for multinomial sampling
 */
export function buildMultinomialShader(params: MultinomialShaderParams): string {
    const { numClasses, replacement } = params;

    const parts: string[] = [];

    // 1. Philox core code for random number generation
    parts.push(PHILOX_WGSL_CORE);

    // 2. Uniforms declaration
    parts.push(MULTINOMIAL_UNIFORMS_WGSL);

    // 3. Buffer declarations
    parts.push(/* wgsl */`
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<i32>;
`);

    // 4. Shared memory for CDF (workgroup-local)
    // We use a fixed workgroup size and shared memory size
    parts.push(/* wgsl */`
// Workgroup shared memory for CDF computation
var<workgroup> shared_cdf: array<f32, ${Math.min(numClasses, 1024)}>;
var<workgroup> shared_total: f32;
`);

    // 5. Main compute shader
    parts.push(generateMainFunction(replacement));

    return parts.join('\n');
}

/**
 * Generate the main compute shader function for multinomial
 */
function generateMainFunction(replacement: boolean): string {
    return /* wgsl */`
// Binary search to find the index where cdf[index-1] < u <= cdf[index]
fn binary_search_cdf(start_idx: u32, num_classes: u32, u: f32) -> i32 {
    var left = 0u;
    var right = num_classes;
    
    while (left < right) {
        let mid = left + (right - left) / 2u;
        let cdf_val = input[start_idx + mid];
        if (cdf_val < u) {
            left = mid + 1u;
        } else {
            right = mid;
        }
    }
    
    return i32(min(left, num_classes - 1u));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_id = gid.x;
    
    // Each thread handles one sample for one batch item
    let total_samples = uniforms.batch_size * uniforms.num_samples;
    if (thread_id >= total_samples) {
        return;
    }
    
    // Determine which batch and which sample this thread handles
    let batch_idx = thread_id / uniforms.num_samples;
    let sample_idx = thread_id % uniforms.num_samples;
    
    // Input row start for this batch
    let input_start = uniforms.input_offset + batch_idx * uniforms.num_classes;
    
    // Output position
    let output_idx = uniforms.output_offset + batch_idx * uniforms.num_samples + sample_idx;
    
    // First, compute the sum for normalization
    // This is done per-thread for simplicity (each thread reads its row)
    var sum = 0.0;
    for (var i = 0u; i < uniforms.num_classes; i = i + 1u) {
        sum = sum + input[input_start + i];
    }
    
    // Generate random number for this sample
    let counter = vec4<u32>(thread_id + uniforms.base_offset, 0u, 0u, 0u);
    let key = vec2<u32>(uniforms.key0, uniforms.key1);
    let random4 = philox4x32_10(counter, key);
    let u = u32_to_uniform(random4.x);
    
    // Scale u by sum (so we can work with unnormalized probabilities)
    let threshold = u * sum;
    
    // Binary search: find the smallest index i such that cumsum(probs[0:i+1]) > threshold
    var cumsum = 0.0;
    var selected_idx = 0i;
    for (var i = 0u; i < uniforms.num_classes; i = i + 1u) {
        cumsum = cumsum + input[input_start + i];
        if (cumsum > threshold) {
            selected_idx = i32(i);
            break;
        }
    }
    
    // Handle edge case: if loop finished without break, select last valid index
    if (cumsum <= threshold) {
        selected_idx = i32(uniforms.num_classes - 1u);
    }
    
    output[output_idx] = selected_idx;
}
`;
}

/**
 * Build shader for computing CDF (separate kernel for larger inputs)
 * This is used when numClasses is large and we need to precompute CDF
 */
export function buildCdfShader(): string {
    return /* wgsl */`
struct CdfUniforms {
    batch_size: u32,
    num_classes: u32,
    input_offset: u32,
    output_offset: u32,
};

@group(0) @binding(0) var<uniform> uniforms: CdfUniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.x;
    
    if (batch_idx >= uniforms.batch_size) {
        return;
    }
    
    let input_start = uniforms.input_offset + batch_idx * uniforms.num_classes;
    let output_start = uniforms.output_offset + batch_idx * uniforms.num_classes;
    
    // Compute CDF via sequential prefix sum
    var cumsum = 0.0;
    for (var i = 0u; i < uniforms.num_classes; i = i + 1u) {
        cumsum = cumsum + input[input_start + i];
        output[output_start + i] = cumsum;
    }
}
`;
}
