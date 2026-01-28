/**
 * NanMean Shader Builder
 * 
 * Generates WGSL shaders for nanmean reduction
 * 
 * Logic:
 * - Accumulate {sum, count} pair
 * - Ignore NaN values (add 0 to sum, 0 to count)
 * - Final result = sum / count
 */

import type { ITensorIterator } from '@kandle/types';
import { getComputeType, generateCastSnippet } from '../../base/dtype';
import type { WgslDType } from '../../types';

// ============================================================================
// Core WGSL Snippets
// ============================================================================

function generateNanMeanStruct(computeType: string): string {
    return `
struct NanMeanState {
    sum: ${computeType},
    count: u32,
}
`;
}

function generateNanMeanInit(computeType: string): string {
    return `
fn nm_init(val: ${computeType}) -> NanMeanState {
    // Check NaN using val != val
    if (val != val) {
        return NanMeanState(${computeType}(0), 0u);
    }
    return NanMeanState(val, 1u);
}

fn nm_empty() -> NanMeanState {
    return NanMeanState(${computeType}(0), 0u);
}
`;
}

function generateNanMeanCombine(computeType: string): string {
    return `
fn nm_combine(a: NanMeanState, b: NanMeanState) -> NanMeanState {
    return NanMeanState(a.sum + b.sum, a.count + b.count);
}
`;
}

function generateNanMeanFinalize(computeType: WgslDType, outputType: WgslDType): string {
    const castToOutput = computeType !== outputType
        ? generateCastSnippet('result', computeType, outputType)
        : 'result';

    return `
fn nm_finalize(state: NanMeanState) -> ${outputType} {
    if (state.count == 0u) {
        // Return NaN. 0/0 is typically NaN in float.
        let zero = ${computeType}(0);
        let result = zero / zero; 
        return ${castToOutput};
    }
    let result = state.sum / ${computeType}(state.count);
    return ${castToOutput};
}
`;
}

// ============================================================================
// Dimensional Reduction Shader
// ============================================================================

export function buildDimNanMeanShader(
    iter: ITensorIterator,
    workgroupSize: number
): string {
    const inputDtype = iter.input(0).dtype;
    const outputDtype = iter.output(0).dtype;
    const computeType = getComputeType(inputDtype);
    const outputType = getComputeType(outputDtype);

    const rank = iter.inputShape.length;
    const reductionRank = iter.reductionShape.length;

    const struct = generateNanMeanStruct(computeType);
    const init = generateNanMeanInit(computeType);
    const combine = generateNanMeanCombine(computeType);
    const finalize = generateNanMeanFinalize(computeType, outputType);

    const inputToCompute = generateCastSnippet('raw_val', getComputeType(inputDtype), computeType);

    return `
${struct}
${init}
${combine}
${finalize}

struct Uniforms {
    outputNumel: u32,
    reductionNumel: u32,
    padding0: u32,
    rank: u32,
    inputShape: vec4<u32>,
    inputShape2: vec4<u32>,
    outputShape: vec4<u32>,
    outputShape2: vec4<u32>,
    inputStrides: vec4<u32>,
    inputStrides2: vec4<u32>,
    reductionStrides: vec4<u32>,
    reductionStrides2: vec4<u32>,
    reductionShape: vec4<u32>,
    reductionShape2: vec4<u32>,
    inputOffset: u32,
    outputOffset: u32,
}

@group(0) @binding(0) var<storage, read> input: array<${getComputeType(inputDtype)}>;
@group(0) @binding(1) var<storage, read_write> output: array<${outputType}>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

var<workgroup> shared_mem: array<NanMeanState, ${workgroupSize}>;

@compute @workgroup_size(${workgroupSize})
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let output_idx = workgroup_id.x;
    let thread_id = local_id.x;
    
    if (output_idx >= uniforms.outputNumel) {
        return;
    }
    
    // Calc base offset
    var base_offset = uniforms.inputOffset;
    {
        var remaining = output_idx;
        ${generateOffsetCalculation('base_offset', 'remaining', 'inputStrides', 'outputShape', rank)}
    }
    
    var local_state = nm_empty();
    let stride = ${workgroupSize}u;
    
    for (var r = thread_id; r < uniforms.reductionNumel; r += stride) {
        var red_offset = 0u;
        {
            var remaining = r;
            ${generateReductionOffsetCalculation('red_offset', 'remaining', 'reductionStrides', 'reductionShape', reductionRank)}
        }
        
        let input_idx = base_offset + red_offset;
        let raw_val = input[input_idx];
        let val = ${inputToCompute};
        local_state = nm_combine(local_state, nm_init(val));
    }
    
    shared_mem[thread_id] = local_state;
    workgroupBarrier();
    
    for (var s = ${workgroupSize}u / 2u; s > 0u; s >>= 1u) {
        if (thread_id < s) {
            shared_mem[thread_id] = nm_combine(shared_mem[thread_id], shared_mem[thread_id + s]);
        }
        workgroupBarrier();
    }
    
    if (thread_id == 0u) {
        output[uniforms.outputOffset + output_idx] = nm_finalize(shared_mem[0]);
    }
}
`;
}

// ============================================================================
// Global Reduction Shaders
// ============================================================================

export function buildNaiveGlobalNanMeanShader(
    iter: ITensorIterator,
    workgroupSize: number
): string {
    const inputDtype = iter.input(0).dtype;
    const outputDtype = iter.output(0).dtype;
    const computeType = getComputeType(inputDtype);
    const outputType = getComputeType(outputDtype);

    const struct = generateNanMeanStruct(computeType);
    const init = generateNanMeanInit(computeType);
    const combine = generateNanMeanCombine(computeType);
    const finalize = generateNanMeanFinalize(computeType, outputType);
    const inputToCompute = generateCastSnippet('raw_val', getComputeType(inputDtype), computeType);

    return `
${struct}
${init}
${combine}
${finalize}

struct Uniforms {
    numel: u32,
    inputOffset: u32,
    outputOffset: u32,
}

@group(0) @binding(0) var<storage, read> input: array<${getComputeType(inputDtype)}>;
@group(0) @binding(1) var<storage, read_write> output: array<${outputType}>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

var<workgroup> shared_mem: array<NanMeanState, ${workgroupSize}>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let thread_id = local_id.x;
    let stride = ${workgroupSize}u;
    
    var local_state = nm_empty();
    for (var i = thread_id; i < uniforms.numel; i += stride) {
        let raw_val = input[uniforms.inputOffset + i];
        let val = ${inputToCompute};
        local_state = nm_combine(local_state, nm_init(val));
    }
    
    shared_mem[thread_id] = local_state;
    workgroupBarrier();
    
    for (var s = ${workgroupSize}u / 2u; s > 0u; s >>= 1u) {
        if (thread_id < s) {
            shared_mem[thread_id] = nm_combine(shared_mem[thread_id], shared_mem[thread_id + s]);
        }
        workgroupBarrier();
    }
    
    if (thread_id == 0u) {
        output[uniforms.outputOffset] = nm_finalize(shared_mem[0]);
    }
}
`;
}

export function buildStridedNaiveGlobalNanMeanShader(
    iter: ITensorIterator,
    workgroupSize: number,
    rank: number
): string {
    const inputDtype = iter.input(0).dtype;
    const outputDtype = iter.output(0).dtype;
    const computeType = getComputeType(inputDtype);
    const outputType = getComputeType(outputDtype);

    const struct = generateNanMeanStruct(computeType);
    const init = generateNanMeanInit(computeType);
    const combine = generateNanMeanCombine(computeType);
    const finalize = generateNanMeanFinalize(computeType, outputType);
    const inputToCompute = generateCastSnippet('raw_val', getComputeType(inputDtype), computeType);

    return `
${struct}
${init}
${combine}
${finalize}

struct Uniforms {
    numel: u32,
    rank: u32,
    inputOffset: u32,
    outputOffset: u32,
    shape: vec4<u32>,
    shape2: vec4<u32>,
    strides: vec4<u32>,
    strides2: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> input: array<${getComputeType(inputDtype)}>;
@group(0) @binding(1) var<storage, read_write> output: array<${outputType}>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

var<workgroup> shared_mem: array<NanMeanState, ${workgroupSize}>;

fn compute_offset(logical_idx: u32) -> u32 {
    var offset = uniforms.inputOffset;
    var remaining = logical_idx;
    ${generateStridedOffsetCode(rank)}
    return offset;
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let thread_id = local_id.x;
    let stride = ${workgroupSize}u;
    
    var local_state = nm_empty();
    for (var i = thread_id; i < uniforms.numel; i += stride) {
        let physical_idx = compute_offset(i);
        let raw_val = input[physical_idx];
        let val = ${inputToCompute};
        local_state = nm_combine(local_state, nm_init(val));
    }
    
    shared_mem[thread_id] = local_state;
    workgroupBarrier();
    
    for (var s = ${workgroupSize}u / 2u; s > 0u; s >>= 1u) {
        if (thread_id < s) {
            shared_mem[thread_id] = nm_combine(shared_mem[thread_id], shared_mem[thread_id + s]);
        }
        workgroupBarrier();
    }
    
    if (thread_id == 0u) {
        output[uniforms.outputOffset] = nm_finalize(shared_mem[0]);
    }
}
`;
}

export function buildGlobalNanMeanStage1Shader(
    iter: ITensorIterator,
    workgroupSize: number
): string {
    const inputDtype = iter.input(0).dtype;
    const computeType = getComputeType(inputDtype);

    const struct = generateNanMeanStruct(computeType);
    const init = generateNanMeanInit(computeType);
    const combine = generateNanMeanCombine(computeType);
    const inputToCompute = generateCastSnippet('raw_val', getComputeType(inputDtype), computeType);

    return `
${struct}
${init}
${combine}

struct Uniforms {
    numel: u32,
    inputOffset: u32,
}

@group(0) @binding(0) var<storage, read> input: array<${getComputeType(inputDtype)}>;
@group(0) @binding(1) var<storage, read_write> partial_results: array<NanMeanState>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

var<workgroup> shared_mem: array<NanMeanState, ${workgroupSize}>;

@compute @workgroup_size(${workgroupSize})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let global_idx = global_id.x;
    let thread_id = local_id.x;
    let workgroup_idx = workgroup_id.x;
    let total_threads = num_workgroups.x * ${workgroupSize}u;
    
    var local_state = nm_empty();
    for (var i = global_idx; i < uniforms.numel; i += total_threads) {
        let raw_val = input[uniforms.inputOffset + i];
        let val = ${inputToCompute};
        local_state = nm_combine(local_state, nm_init(val));
    }
    
    shared_mem[thread_id] = local_state;
    workgroupBarrier();
    
    for (var s = ${workgroupSize}u / 2u; s > 0u; s >>= 1u) {
        if (thread_id < s) {
            shared_mem[thread_id] = nm_combine(shared_mem[thread_id], shared_mem[thread_id + s]);
        }
        workgroupBarrier();
    }
    
    if (thread_id == 0u) {
        partial_results[workgroup_idx] = shared_mem[0];
    }
}
`;
}

export function buildGlobalNanMeanStage2Shader(
    computeType: WgslDType,
    outputType: WgslDType,
    workgroupSize: number
): string {
    const struct = generateNanMeanStruct(computeType);
    const init = generateNanMeanInit(computeType); // Need empty
    const combine = generateNanMeanCombine(computeType);
    const finalize = generateNanMeanFinalize(computeType, outputType);

    return `
${struct}
${init}
${combine}
${finalize}

struct Uniforms {
    numPartials: u32,
    outputOffset: u32,
}

@group(0) @binding(0) var<storage, read> partial_results: array<NanMeanState>;
@group(0) @binding(1) var<storage, read_write> output: array<${outputType}>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

var<workgroup> shared_mem: array<NanMeanState, ${workgroupSize}>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let thread_id = local_id.x;
    let stride = ${workgroupSize}u;
    
    var local_state = nm_empty();
    for (var i = thread_id; i < uniforms.numPartials; i += stride) {
        local_state = nm_combine(local_state, partial_results[i]);
    }
    
    shared_mem[thread_id] = local_state;
    workgroupBarrier();
    
    for (var s = ${workgroupSize}u / 2u; s > 0u; s >>= 1u) {
        if (thread_id < s) {
            shared_mem[thread_id] = nm_combine(shared_mem[thread_id], shared_mem[thread_id + s]);
        }
        workgroupBarrier();
    }
    
    if (thread_id == 0u) {
        output[uniforms.outputOffset] = nm_finalize(shared_mem[0]);
    }
}
`;
}


// Helpers (copied to avoid dependency loop or complexity)
function generateOffsetCalculation(
    offsetVar: string,
    remainingVar: string,
    stridesField: string,
    shapeField: string,
    rank: number
): string {
    const lines: string[] = [];
    for (let i = 0; i < rank; i++) {
        const field = i < 4 ? stridesField : `${stridesField}2`;
        const shapeF = i < 4 ? shapeField : `${shapeField}2`;
        const component = ['x', 'y', 'z', 'w'][i % 4];

        lines.push(`{
            let dim_size = uniforms.${shapeF}.${component};
            let stride = uniforms.${field}.${component};
            if (dim_size > 0u) {
                let coord = ${remainingVar} % dim_size;
                ${remainingVar} = ${remainingVar} / dim_size;
                ${offsetVar} += coord * stride;
            }
        }`);
    }
    return lines.join('\n        ');
}

function generateReductionOffsetCalculation(
    offsetVar: string,
    remainingVar: string,
    stridesField: string,
    shapeField: string,
    reductionRank: number
): string {
    const lines: string[] = [];
    for (let i = 0; i < reductionRank; i++) {
        const field = i < 4 ? stridesField : `${stridesField}2`;
        const shapeF = i < 4 ? shapeField : `${shapeField}2`;
        const component = ['x', 'y', 'z', 'w'][i % 4];

        lines.push(`{
            let dim_size = uniforms.${shapeF}.${component};
            let stride = uniforms.${field}.${component};
            if (dim_size > 0u) {
                let coord = ${remainingVar} % dim_size;
                ${remainingVar} = ${remainingVar} / dim_size;
                ${offsetVar} += coord * stride;
            }
        }`);
    }
    return lines.join('\n            ');
}

function generateStridedOffsetCode(rank: number): string {
    const lines: string[] = [];
    for (let i = rank - 1; i >= 0; i--) {
        const shapeField = i < 4 ? 'shape' : 'shape2';
        const stridesField = i < 4 ? 'strides' : 'strides2';
        const component = ['x', 'y', 'z', 'w'][i % 4];

        lines.push(`{
        let dim_size = uniforms.${shapeField}.${component};
        let stride = uniforms.${stridesField}.${component};
        let coord = remaining % dim_size;
        remaining = remaining / dim_size;
        offset += coord * stride;
    }`);
    }
    return lines.join('\n    ');
}
