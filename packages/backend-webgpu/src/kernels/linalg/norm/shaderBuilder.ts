/**
 * Norm Shader Builder
 * 
 * 生成 Lp 范数的 WGSL shader
 * 
 * 优化策略:
 * - p=2 (L2): sqrt(sum(x²)) - 最常用，直接计算
 * - p=1 (L1): sum(|x|) - 直接计算
 * - p=inf: max(|x|) - 使用 max reduction
 * - p=-inf: min(|x|) - 使用 min reduction
 * - p=0: count(x != 0) - 非零计数
 * - 通用 p: (sum(|x|^p))^(1/p) - two-pass
 * 
 * 参考: PyTorch ATen/native/LinearAlgebra.cpp
 */

import type { ITensorIterator } from '@kandle/types';
import { getComputeType, generateCastSnippet } from '../../../base/dtype';
import type { WgslDType } from '../../../types';
import { Logger } from '@kandle/utils';
import { getNormType, type NormOrd } from './types';

const logger = new Logger('Norm-ShaderBuilder');

// ============================================================================
// L2 Norm Shader (最常用)
// ============================================================================

/**
 * 构建 L2 范数 Dimensional Reduction Shader
 * 公式: sqrt(sum(x²))
 */
export function buildDimL2NormShader(
    iter: ITensorIterator,
    workgroupSize: number
): string {
    const inputDtype = iter.input(0).dtype;
    const outputDtype = iter.output(0).dtype;
    const computeType = getComputeType(inputDtype);
    const outputType = getComputeType(outputDtype);

    const rank = iter.inputShape.length;
    const reductionRank = iter.reductionShape.length;

    const inputToCompute = generateCastSnippet('raw_val', getComputeType(inputDtype), computeType);
    const castToOutput = computeType !== outputType
        ? generateCastSnippet('result', computeType, outputType)
        : 'result';

    return `
// ============================================================================
// L2 Norm Dimensional Reduction: sqrt(sum(x²))
// ============================================================================

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

var<workgroup> shared_mem: array<${computeType}, ${workgroupSize}>;

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
    
    // 计算基础输入偏移
    var base_offset = uniforms.inputOffset;
    {
        var remaining = output_idx;
        ${generateOffsetCalculation('base_offset', 'remaining', 'inputStrides', 'outputShape', rank)}
    }
    
    // 累积 x²
    var local_sum = ${computeType}(0);
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
        local_sum += val * val;  // x²
    }
    
    // Tree reduction
    shared_mem[thread_id] = local_sum;
    workgroupBarrier();
    
    for (var s = ${workgroupSize}u / 2u; s > 0u; s >>= 1u) {
        if (thread_id < s) {
            shared_mem[thread_id] += shared_mem[thread_id + s];
        }
        workgroupBarrier();
    }
    
    // sqrt 并写入结果
    if (thread_id == 0u) {
        let result = sqrt(shared_mem[0]);
        output[uniforms.outputOffset + output_idx] = ${castToOutput};
    }
}
`;
}

// ============================================================================
// L1 Norm Shader
// ============================================================================

/**
 * 构建 L1 范数 Dimensional Reduction Shader
 * 公式: sum(|x|)
 */
export function buildDimL1NormShader(
    iter: ITensorIterator,
    workgroupSize: number
): string {
    const inputDtype = iter.input(0).dtype;
    const outputDtype = iter.output(0).dtype;
    const computeType = getComputeType(inputDtype);
    const outputType = getComputeType(outputDtype);

    const rank = iter.inputShape.length;
    const reductionRank = iter.reductionShape.length;

    const inputToCompute = generateCastSnippet('raw_val', getComputeType(inputDtype), computeType);
    const castToOutput = computeType !== outputType
        ? generateCastSnippet('result', computeType, outputType)
        : 'result';

    return `
// ============================================================================
// L1 Norm Dimensional Reduction: sum(|x|)
// ============================================================================

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

var<workgroup> shared_mem: array<${computeType}, ${workgroupSize}>;

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
    
    var base_offset = uniforms.inputOffset;
    {
        var remaining = output_idx;
        ${generateOffsetCalculation('base_offset', 'remaining', 'inputStrides', 'outputShape', rank)}
    }
    
    // 累积 |x|
    var local_sum = ${computeType}(0);
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
        local_sum += abs(val);
    }
    
    // Tree reduction
    shared_mem[thread_id] = local_sum;
    workgroupBarrier();
    
    for (var s = ${workgroupSize}u / 2u; s > 0u; s >>= 1u) {
        if (thread_id < s) {
            shared_mem[thread_id] += shared_mem[thread_id + s];
        }
        workgroupBarrier();
    }
    
    if (thread_id == 0u) {
        let result = shared_mem[0];
        output[uniforms.outputOffset + output_idx] = ${castToOutput};
    }
}
`;
}

// ============================================================================
// Inf Norm Shader (Max Norm)
// ============================================================================

/**
 * 构建 Inf 范数 Dimensional Reduction Shader
 * 公式: max(|x|)
 */
export function buildDimInfNormShader(
    iter: ITensorIterator,
    workgroupSize: number,
    isNegInf: boolean = false
): string {
    const inputDtype = iter.input(0).dtype;
    const outputDtype = iter.output(0).dtype;
    const computeType = getComputeType(inputDtype);
    const outputType = getComputeType(outputDtype);

    const rank = iter.inputShape.length;
    const reductionRank = iter.reductionShape.length;

    const inputToCompute = generateCastSnippet('raw_val', getComputeType(inputDtype), computeType);
    const castToOutput = computeType !== outputType
        ? generateCastSnippet('result', computeType, outputType)
        : 'result';

    // inf: max, -inf: min
    const initVal = isNegInf ? `${computeType}(1e38)` : `${computeType}(0)`;
    const reduceOp = isNegInf ? 'min' : 'max';

    return `
// ============================================================================
// ${isNegInf ? '-Inf' : 'Inf'} Norm Dimensional Reduction: ${isNegInf ? 'min' : 'max'}(|x|)
// ============================================================================

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

var<workgroup> shared_mem: array<${computeType}, ${workgroupSize}>;

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
    
    var base_offset = uniforms.inputOffset;
    {
        var remaining = output_idx;
        ${generateOffsetCalculation('base_offset', 'remaining', 'inputStrides', 'outputShape', rank)}
    }
    
    var local_result = ${initVal};
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
        local_result = ${reduceOp}(local_result, abs(val));
    }
    
    // Tree reduction
    shared_mem[thread_id] = local_result;
    workgroupBarrier();
    
    for (var s = ${workgroupSize}u / 2u; s > 0u; s >>= 1u) {
        if (thread_id < s) {
            shared_mem[thread_id] = ${reduceOp}(shared_mem[thread_id], shared_mem[thread_id + s]);
        }
        workgroupBarrier();
    }
    
    if (thread_id == 0u) {
        let result = shared_mem[0];
        output[uniforms.outputOffset + output_idx] = ${castToOutput};
    }
}
`;
}

// ============================================================================
// Zero Norm Shader (Count Non-Zero)
// ============================================================================

/**
 * 构建 L0 范数 Dimensional Reduction Shader
 * 公式: count(x != 0)
 */
export function buildDimZeroNormShader(
    iter: ITensorIterator,
    workgroupSize: number
): string {
    const inputDtype = iter.input(0).dtype;
    const outputDtype = iter.output(0).dtype;
    const computeType = getComputeType(inputDtype);
    const outputType = getComputeType(outputDtype);

    const rank = iter.inputShape.length;
    const reductionRank = iter.reductionShape.length;

    const inputToCompute = generateCastSnippet('raw_val', getComputeType(inputDtype), computeType);
    const castToOutput = 'f32' !== outputType
        ? generateCastSnippet('result', 'f32', outputType)
        : 'result';

    return `
// ============================================================================
// L0 Norm Dimensional Reduction: count(x != 0)
// ============================================================================

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

var<workgroup> shared_mem: array<f32, ${workgroupSize}>;

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
    
    var base_offset = uniforms.inputOffset;
    {
        var remaining = output_idx;
        ${generateOffsetCalculation('base_offset', 'remaining', 'inputStrides', 'outputShape', rank)}
    }
    
    var local_count = f32(0);
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
        if (val != ${computeType}(0)) {
            local_count += f32(1);
        }
    }
    
    // Tree reduction
    shared_mem[thread_id] = local_count;
    workgroupBarrier();
    
    for (var s = ${workgroupSize}u / 2u; s > 0u; s >>= 1u) {
        if (thread_id < s) {
            shared_mem[thread_id] += shared_mem[thread_id + s];
        }
        workgroupBarrier();
    }
    
    if (thread_id == 0u) {
        let result = shared_mem[0];
        output[uniforms.outputOffset + output_idx] = ${castToOutput};
    }
}
`;
}

// ============================================================================
// General Lp Norm Shader
// ============================================================================

/**
 * 构建通用 Lp 范数 Dimensional Reduction Shader
 * 公式: (sum(|x|^p))^(1/p)
 */
export function buildDimGeneralNormShader(
    iter: ITensorIterator,
    workgroupSize: number
): string {
    const inputDtype = iter.input(0).dtype;
    const outputDtype = iter.output(0).dtype;
    const computeType = getComputeType(inputDtype);
    const outputType = getComputeType(outputDtype);

    const rank = iter.inputShape.length;
    const reductionRank = iter.reductionShape.length;

    const inputToCompute = generateCastSnippet('raw_val', getComputeType(inputDtype), computeType);
    const castToOutput = computeType !== outputType
        ? generateCastSnippet('result', computeType, outputType)
        : 'result';

    return `
// ============================================================================
// General Lp Norm Dimensional Reduction: (sum(|x|^p))^(1/p)
// ============================================================================

struct Uniforms {
    outputNumel: u32,
    reductionNumel: u32,
    p: f32,              // p 值
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

var<workgroup> shared_mem: array<${computeType}, ${workgroupSize}>;

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
    
    let p = ${computeType}(uniforms.p);
    
    var base_offset = uniforms.inputOffset;
    {
        var remaining = output_idx;
        ${generateOffsetCalculation('base_offset', 'remaining', 'inputStrides', 'outputShape', rank)}
    }
    
    // 累积 |x|^p
    var local_sum = ${computeType}(0);
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
        local_sum += pow(abs(val), p);
    }
    
    // Tree reduction
    shared_mem[thread_id] = local_sum;
    workgroupBarrier();
    
    for (var s = ${workgroupSize}u / 2u; s > 0u; s >>= 1u) {
        if (thread_id < s) {
            shared_mem[thread_id] += shared_mem[thread_id + s];
        }
        workgroupBarrier();
    }
    
    // pow(sum, 1/p)
    if (thread_id == 0u) {
        let sum_p = shared_mem[0];
        let result = pow(sum_p, ${computeType}(1) / p);
        output[uniforms.outputOffset + output_idx] = ${castToOutput};
    }
}
`;
}

// ============================================================================
// Naive Global Norm Shaders
// ============================================================================

/**
 * 构建 Naive Global L2 Norm Shader
 */
export function buildNaiveGlobalL2NormShader(
    iter: ITensorIterator,
    workgroupSize: number
): string {
    const inputDtype = iter.input(0).dtype;
    const outputDtype = iter.output(0).dtype;
    const computeType = getComputeType(inputDtype);
    const outputType = getComputeType(outputDtype);

    const inputToCompute = generateCastSnippet('raw_val', getComputeType(inputDtype), computeType);
    const castToOutput = computeType !== outputType
        ? generateCastSnippet('result', computeType, outputType)
        : 'result';

    return `
// ============================================================================
// Global L2 Norm (Naive): sqrt(sum(x²))
// ============================================================================

struct Uniforms {
    numel: u32,
    inputOffset: u32,
    outputOffset: u32,
}

@group(0) @binding(0) var<storage, read> input: array<${getComputeType(inputDtype)}>;
@group(0) @binding(1) var<storage, read_write> output: array<${outputType}>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

var<workgroup> shared_mem: array<${computeType}, ${workgroupSize}>;

@compute @workgroup_size(${workgroupSize})
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let thread_id = local_id.x;
    let stride = ${workgroupSize}u;
    
    var local_sum = ${computeType}(0);
    for (var i = thread_id; i < uniforms.numel; i += stride) {
        let raw_val = input[uniforms.inputOffset + i];
        let val = ${inputToCompute};
        local_sum += val * val;
    }
    
    shared_mem[thread_id] = local_sum;
    workgroupBarrier();
    
    for (var s = ${workgroupSize}u / 2u; s > 0u; s >>= 1u) {
        if (thread_id < s) {
            shared_mem[thread_id] += shared_mem[thread_id + s];
        }
        workgroupBarrier();
    }
    
    if (thread_id == 0u) {
        let result = sqrt(shared_mem[0]);
        output[uniforms.outputOffset] = ${castToOutput};
    }
}
`;
}

/**
 * 构建 Naive Global Norm Shader (支持所有 normType)
 */
export function buildNaiveGlobalNormShader(
    iter: ITensorIterator,
    workgroupSize: number,
    normType: 'zero' | 'one' | 'two' | 'inf' | 'neg_inf' | 'general',
    p: NormOrd
): string {
    const inputDtype = iter.input(0).dtype;
    const outputDtype = iter.output(0).dtype;
    const computeType = getComputeType(inputDtype);
    const outputType = getComputeType(outputDtype);

    const inputToCompute = generateCastSnippet('raw_val', getComputeType(inputDtype), computeType);
    const castToOutput = computeType !== outputType
        ? generateCastSnippet('result', computeType, outputType)
        : 'result';

    // 根据 normType 生成不同的累积和最终化逻辑
    let initVal: string;
    let accumOp: string;
    let reduceOp: string;
    let finalOp: string;

    switch (normType) {
        case 'zero':
            // L0: count(x != 0)
            initVal = `${computeType}(0)`;
            accumOp = `if (val != ${computeType}(0)) { local_acc += ${computeType}(1); }`;
            reduceOp = `shared_mem[thread_id] += shared_mem[thread_id + s];`;
            finalOp = `let result = shared_mem[0];`;
            break;
        case 'one':
            // L1: sum(|x|)
            initVal = `${computeType}(0)`;
            accumOp = `local_acc += abs(val);`;
            reduceOp = `shared_mem[thread_id] += shared_mem[thread_id + s];`;
            finalOp = `let result = shared_mem[0];`;
            break;
        case 'two':
            // L2: sqrt(sum(x²))
            initVal = `${computeType}(0)`;
            accumOp = `local_acc += val * val;`;
            reduceOp = `shared_mem[thread_id] += shared_mem[thread_id + s];`;
            finalOp = `let result = sqrt(shared_mem[0]);`;
            break;
        case 'inf':
            // Inf: max(|x|)
            initVal = `${computeType}(0)`;
            accumOp = `local_acc = max(local_acc, abs(val));`;
            reduceOp = `shared_mem[thread_id] = max(shared_mem[thread_id], shared_mem[thread_id + s]);`;
            finalOp = `let result = shared_mem[0];`;
            break;
        case 'neg_inf':
            // -Inf: min(|x|)
            initVal = `${computeType}(1e38)`;
            accumOp = `local_acc = min(local_acc, abs(val));`;
            reduceOp = `shared_mem[thread_id] = min(shared_mem[thread_id], shared_mem[thread_id + s]);`;
            finalOp = `let result = shared_mem[0];`;
            break;
        case 'general':
        default:
            // General Lp: (sum(|x|^p))^(1/p)
            initVal = `${computeType}(0)`;
            accumOp = `local_acc += pow(abs(val), p);`;
            reduceOp = `shared_mem[thread_id] += shared_mem[thread_id + s];`;
            finalOp = `let result = pow(shared_mem[0], ${computeType}(1) / p);`;
            break;
    }

    return `
// ============================================================================
// Global Norm (Naive): ${normType}
// ============================================================================

struct Uniforms {
    numel: u32,
    inputOffset: u32,
    outputOffset: u32,
    p: f32,  // p 值 (for general norm)
}

@group(0) @binding(0) var<storage, read> input: array<${getComputeType(inputDtype)}>;
@group(0) @binding(1) var<storage, read_write> output: array<${outputType}>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

var<workgroup> shared_mem: array<${computeType}, ${workgroupSize}>;

@compute @workgroup_size(${workgroupSize})
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let thread_id = local_id.x;
    let stride = ${workgroupSize}u;
    ${normType === 'general' ? `let p = ${computeType}(uniforms.p);` : ''}
    
    var local_acc = ${initVal};
    for (var i = thread_id; i < uniforms.numel; i += stride) {
        let raw_val = input[uniforms.inputOffset + i];
        let val = ${inputToCompute};
        ${accumOp}
    }
    
    shared_mem[thread_id] = local_acc;
    workgroupBarrier();
    
    for (var s = ${workgroupSize}u / 2u; s > 0u; s >>= 1u) {
        if (thread_id < s) {
            ${reduceOp}
        }
        workgroupBarrier();
    }
    
    if (thread_id == 0u) {
        ${finalOp}
        output[uniforms.outputOffset] = ${castToOutput};
    }
}
`;
}

/**
 * 根据 normType 选择合适的 dimensional shader
 */
export function buildDimNormShader(
    iter: ITensorIterator,
    workgroupSize: number,
    normType: 'zero' | 'one' | 'two' | 'inf' | 'neg_inf' | 'general'
): string {
    switch (normType) {
        case 'zero':
            return buildDimZeroNormShader(iter, workgroupSize);
        case 'one':
            return buildDimL1NormShader(iter, workgroupSize);
        case 'two':
            return buildDimL2NormShader(iter, workgroupSize);
        case 'inf':
            return buildDimInfNormShader(iter, workgroupSize, false);
        case 'neg_inf':
            return buildDimInfNormShader(iter, workgroupSize, true);
        case 'general':
        default:
            return buildDimGeneralNormShader(iter, workgroupSize);
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

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
