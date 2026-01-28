/**
 * Random Shader Builder
 * 
 * Generates WGSL shaders for rand, randn, and randint operations
 */

import type { DType } from '@kandle/types';
import { PHILOX_WGSL_CORE, RANDOM_UNIFORMS_WGSL } from './philox.wgsl';
import type { RandomOpType, RandomShaderParams } from './types';

/**
 * Get WGSL type from DType
 */
function wgslType(dtype: DType): string {
    switch (dtype) {
        case 'float32': return 'f32';
        case 'float64': return 'f32';   // WebGPU 不支持 f64
        case 'int32': return 'i32';
        case 'int64': return 'i32';     // WebGPU 不支持 i64
        case 'uint32': return 'u32';
        case 'int16': return 'i32';
        case 'uint16': return 'u32';
        case 'int8': return 'i32';
        case 'uint8': return 'u32';
        case 'bool': return 'u32';
        default: return 'f32';
    }
}

/**
 * Build the complete shader for a random operation
 */
export function buildRandomShader(params: RandomShaderParams): string {
    const { opType, outputDtype } = params;
    const wgslOutputType = wgslType(outputDtype as DType);

    const parts: string[] = [];

    // 1. Philox core code
    parts.push(PHILOX_WGSL_CORE);

    // 2. Uniforms declaration
    parts.push(RANDOM_UNIFORMS_WGSL);

    // 3. Output buffer declaration
    parts.push(generateOutputBuffer(opType, wgslOutputType));

    // 4. Main compute shader
    parts.push(generateMainFunction(opType, wgslOutputType));

    return parts.join('\n');
}

/**
 * Generate output buffer declaration based on op type
 */
function generateOutputBuffer(opType: RandomOpType, wgslOutputType: string): string {
    // For randint with integer output, use the appropriate type
    // For rand/randn, always use f32
    const bufferType = opType === 'randint' ? wgslOutputType : 'f32';

    return /* wgsl */`
@group(0) @binding(1) var<storage, read_write> output: array<${bufferType}>;
`;
}

/**
 * Generate the main compute shader function
 */
function generateMainFunction(opType: RandomOpType, wgslOutputType: string): string {
    switch (opType) {
        case 'rand':
            return generateRandMain();
        case 'randn':
            return generateRandnMain();
        case 'randint':
            return generateRandintMain(wgslOutputType);
        default:
            throw new Error(`Unknown random operation: ${opType}`);
    }
}

/**
 * Generate main function for rand() - Uniform [0, 1)
 */
function generateRandMain(): string {
    return /* wgsl */`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_id = gid.x;
    
    // Skip out-of-bounds threads
    if (thread_id >= uniforms.numel) {
        return;
    }
    
    // Each Philox call generates 4 u32 values
    // Map thread_id to philox_id and sub_id
    let philox_id = thread_id / 4u;
    let sub_id = thread_id % 4u;
    
    // Construct counter: [philox_id + base_offset, 0, 0, 0]
    let counter = vec4<u32>(philox_id + uniforms.base_offset, 0u, 0u, 0u);
    let key = vec2<u32>(uniforms.key0, uniforms.key1);
    
    // Generate 4 random u32 values
    let random4 = philox4x32_10(counter, key);
    
    // Select the appropriate sub-element
    var random_u32: u32;
    switch (sub_id) {
        case 0u: { random_u32 = random4.x; }
        case 1u: { random_u32 = random4.y; }
        case 2u: { random_u32 = random4.z; }
        default: { random_u32 = random4.w; }
    }
    
    // Convert to [0, 1) and store
    output[thread_id + uniforms.output_offset] = u32_to_uniform(random_u32);
}
`;
}

/**
 * Generate main function for randn() - Normal N(0, 1) using Box-Muller
 */
function generateRandnMain(): string {
    return /* wgsl */`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_id = gid.x;
    
    // Skip out-of-bounds threads
    if (thread_id >= uniforms.numel) {
        return;
    }
    
    // Box-Muller needs 2 uniform values to generate 2 normal values
    // Each Philox call generates 4 u32 → 4 uniforms → 2 pairs
    // So pair_id maps to which pair within the Philox output
    let pair_id = thread_id / 2u;
    let sub_id = thread_id % 2u;
    
    // Construct counter for this pair
    let counter = vec4<u32>(pair_id + uniforms.base_offset, 0u, 0u, 0u);
    let key = vec2<u32>(uniforms.key0, uniforms.key1);
    
    // Generate 4 random u32 values
    let random4 = philox4x32_10(counter, key);
    
    // Select u1, u2 pair based on sub_id
    var u1: f32;
    var u2: f32;
    if (sub_id == 0u) {
        u1 = u32_to_uniform(random4.x);
        u2 = u32_to_uniform(random4.y);
    } else {
        u1 = u32_to_uniform(random4.z);
        u2 = u32_to_uniform(random4.w);
    }
    
    // Box-Muller transform
    let z = box_muller(u1, u2);
    
    output[thread_id + uniforms.output_offset] = z;
}
`;
}

/**
 * Generate main function for randint() - Uniform integers [low, high)
 */
function generateRandintMain(wgslOutputType: string): string {
    // Handle different integer output types
    const castToOutput = wgslOutputType === 'i32' ? '' : `${wgslOutputType}`;

    return /* wgsl */`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_id = gid.x;
    
    // Skip out-of-bounds threads
    if (thread_id >= uniforms.numel) {
        return;
    }
    
    // Each Philox call generates 4 u32 values
    let philox_id = thread_id / 4u;
    let sub_id = thread_id % 4u;
    
    // Construct counter
    let counter = vec4<u32>(philox_id + uniforms.base_offset, 0u, 0u, 0u);
    let key = vec2<u32>(uniforms.key0, uniforms.key1);
    
    // Generate 4 random u32 values
    let random4 = philox4x32_10(counter, key);
    
    // Select the appropriate sub-element
    var random_u32: u32;
    switch (sub_id) {
        case 0u: { random_u32 = random4.x; }
        case 1u: { random_u32 = random4.y; }
        case 2u: { random_u32 = random4.z; }
        default: { random_u32 = random4.w; }
    }
    
    // Map to [low, high)
    let range = u32(uniforms.high - uniforms.low);
    let result = uniforms.low + i32(random_u32 % range);
    
    // Store result (cast to output type if needed)
    output[thread_id + uniforms.output_offset] = ${castToOutput ? `${castToOutput}(result)` : 'result'};
}
`;
}
