/**
 * Welford Shader Builder
 * 
 * 生成 Welford 算法的 WGSL shader
 * 
 * 核心算法:
 * 1. WelfordData 三元组: { mean, m2, n }
 * 2. welford_init(val): 初始化单个值
 * 3. welford_combine(a, b): 合并两个中间状态
 * 4. Tree reduction: 并行归约
 * 5. Finalize: m2 / (n - correction) [+ sqrt]
 */

import type { ITensorIterator } from '@kandle/types';
import { getComputeType, generateCastSnippet } from '../../base/dtype';
import type { WgslDType } from '../../types';
import { Logger } from '@kandle/utils';
import { WELFORD_OPS } from './ops';

const logger = new Logger('Welford-ShaderBuilder');

// ============================================================================
// Core Welford WGSL Snippets
// ============================================================================

/**
 * 生成 WelfordData 结构体定义
 */
function generateWelfordStruct(computeType: string): string {
    return `
// Welford 三元组: 在线均值/方差算法的中间状态
struct WelfordData {
    mean: ${computeType},  // 当前均值
    m2: ${computeType},    // 当前平方差和 sum((x - mean)^2)
    n: ${computeType},     // 当前计数
}
`;
}

/**
 * 生成 welford_init 函数
 */
function generateWelfordInit(computeType: string): string {
    return `
// 初始化: 将单个值转换为 WelfordData
fn welford_init(val: ${computeType}) -> WelfordData {
    return WelfordData(val, ${computeType}(0), ${computeType}(1));
}

// 初始化: 空状态 (用于越界线程)
fn welford_empty() -> WelfordData {
    return WelfordData(${computeType}(0), ${computeType}(0), ${computeType}(0));
}
`;
}

/**
 * 生成 welford_combine 函数
 * 这是 Welford 的精髓 - 数值稳定的合并算法
 */
function generateWelfordCombine(computeType: string): string {
    return `
// 合并两个 WelfordData (并行归约的核心)
// 参考: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
fn welford_combine(a: WelfordData, b: WelfordData) -> WelfordData {
    let n_new = a.n + b.n;
    
    // 处理空状态
    if (n_new == ${computeType}(0)) {
        return WelfordData(${computeType}(0), ${computeType}(0), ${computeType}(0));
    }
    
    // 只有一边有数据
    if (a.n == ${computeType}(0)) { return b; }
    if (b.n == ${computeType}(0)) { return a; }
    
    let delta = b.mean - a.mean;
    
    // 新均值: 加权平均
    let mean_new = a.mean + delta * (b.n / n_new);
    
    // 新 M2: Welford 并行合并公式 (数值稳定)
    // M2_new = M2_a + M2_b + delta^2 * (n_a * n_b / n_new)
    let m2_new = a.m2 + b.m2 + delta * delta * (a.n * b.n / n_new);
    
    return WelfordData(mean_new, m2_new, n_new);
}
`;
}

/**
 * 生成 finalize 函数
 */
function generateWelfordFinalize(computeType: WgslDType, applySqrt: boolean, outputType: WgslDType): string {
    const sqrtLine = applySqrt ? `result = sqrt(result);` : '';
    const castToOutput = computeType !== outputType
        ? generateCastSnippet('result', computeType, outputType)
        : 'result';

    return `
// 计算最终结果: variance 或 std
fn welford_finalize(state: WelfordData, correction: ${computeType}) -> ${outputType} {
    // 避免除以零或负数
    let divisor = max(state.n - correction, ${computeType}(0));
    
    var result: ${computeType};
    if (divisor <= ${computeType}(0)) {
        // n <= correction 时返回 0 (PyTorch 行为)
        result = ${computeType}(0);
    } else {
        result = state.m2 / divisor;
    }
    
    ${sqrtLine}
    
    return ${castToOutput};
}
`;
}

// ============================================================================
// Dimensional Reduction Shader
// ============================================================================

/**
 * 构建 Dimensional Welford Reduction Shader
 * 
 * 每个 workgroup 处理一个输出元素，归约其对应的所有输入元素
 */
export function buildDimWelfordShader(
    iter: ITensorIterator,
    dispatchKey: string,
    workgroupSize: number
): string {
    const config = WELFORD_OPS[dispatchKey];
    if (!config) {
        throw new Error(`Unknown Welford operation: ${dispatchKey}`);
    }

    const inputDtype = iter.input(0).dtype;
    const outputDtype = iter.output(0).dtype;
    const computeType = getComputeType(inputDtype);
    const outputType = getComputeType(outputDtype);

    const rank = iter.inputShape.length;
    const reductionRank = iter.reductionShape.length;

    logger.debug(`Building dim Welford shader: dispatchKey=${dispatchKey}, rank=${rank}, reductionRank=${reductionRank}`);

    // 生成所有 Welford 函数
    const welfordStruct = generateWelfordStruct(computeType);
    const welfordInit = generateWelfordInit(computeType);
    const welfordCombine = generateWelfordCombine(computeType);
    const welfordFinalize = generateWelfordFinalize(computeType, config.applySqrt, outputType);

    // Cast snippets
    const inputToCompute = generateCastSnippet('raw_val', getComputeType(inputDtype), computeType);

    return `
// ============================================================================
// Welford Dimensional Reduction: ${dispatchKey}
// ============================================================================

${welfordStruct}

${welfordInit}

${welfordCombine}

${welfordFinalize}

// Uniform buffer layout
struct Uniforms {
    outputNumel: u32,           // 输出元素数
    reductionNumel: u32,        // 每个输出要归约的元素数
    correction: f32,            // 贝塞尔校正
    rank: u32,                  // 输入 rank
    inputShape: vec4<u32>,      // 输入形状 (最多 4 维)
    inputShape2: vec4<u32>,     // 输入形状扩展 (5-8 维)
    outputShape: vec4<u32>,     // 输出形状
    outputShape2: vec4<u32>,    // 输出形状扩展
    inputStrides: vec4<u32>,    // 输入 parallel strides
    inputStrides2: vec4<u32>,   // 输入 parallel strides 扩展
    reductionStrides: vec4<u32>,// 归约维度 strides
    reductionStrides2: vec4<u32>,// 归约维度 strides 扩展
    reductionShape: vec4<u32>,  // 归约维度 shape
    reductionShape2: vec4<u32>, // 归约维度 shape 扩展
    inputOffset: u32,           // 输入偏移
    outputOffset: u32,          // 输出偏移
}

@group(0) @binding(0) var<storage, read> input: array<${getComputeType(inputDtype)}>;
@group(0) @binding(1) var<storage, read_write> output: array<${outputType}>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

var<workgroup> shared_mem: array<WelfordData, ${workgroupSize}>;

@compute @workgroup_size(${workgroupSize})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let output_idx = workgroup_id.x;
    let thread_id = local_id.x;
    
    // 越界检查
    if (output_idx >= uniforms.outputNumel) {
        return;
    }
    
    // Step 1: 计算此输出元素的基础输入偏移
    var base_offset = uniforms.inputOffset;
    {
        var remaining = output_idx;
        ${generateOffsetCalculation('base_offset', 'remaining', 'inputStrides', 'outputShape', rank)}
    }
    
    // Step 2: 每个线程处理多个归约元素 (stride loop)
    var local_state = welford_empty();
    let stride = ${workgroupSize}u;
    
    for (var r = thread_id; r < uniforms.reductionNumel; r += stride) {
        // 计算归约维度内的偏移
        var red_offset = 0u;
        {
            var remaining = r;
            ${generateReductionOffsetCalculation('red_offset', 'remaining', 'reductionStrides', 'reductionShape', reductionRank)}
        }
        
        let input_idx = base_offset + red_offset;
        let raw_val = input[input_idx];
        let val = ${inputToCompute};
        local_state = welford_combine(local_state, welford_init(val));
    }
    
    // Step 3: 存入共享内存
    shared_mem[thread_id] = local_state;
    workgroupBarrier();
    
    // Step 4: Tree reduction in shared memory
    for (var s = ${workgroupSize}u / 2u; s > 0u; s >>= 1u) {
        if (thread_id < s) {
            shared_mem[thread_id] = welford_combine(shared_mem[thread_id], shared_mem[thread_id + s]);
        }
        workgroupBarrier();
    }
    
    // Step 5: 线程 0 写入最终结果
    if (thread_id == 0u) {
        let final_state = shared_mem[0];
        let correction = ${computeType}(uniforms.correction);
        let result = welford_finalize(final_state, correction);
        output[uniforms.outputOffset + output_idx] = result;
    }
}
`;
}

// ============================================================================
// Global Reduction Shaders (Single-pass for small data)
// ============================================================================

/**
 * 构建 Naive Global Welford Shader (单 workgroup)
 */
export function buildNaiveGlobalWelfordShader(
    iter: ITensorIterator,
    dispatchKey: string,
    workgroupSize: number
): string {
    const config = WELFORD_OPS[dispatchKey];
    if (!config) {
        throw new Error(`Unknown Welford operation: ${dispatchKey}`);
    }

    const inputDtype = iter.input(0).dtype;
    const outputDtype = iter.output(0).dtype;
    const computeType = getComputeType(inputDtype);
    const outputType = getComputeType(outputDtype);

    const welfordStruct = generateWelfordStruct(computeType);
    const welfordInit = generateWelfordInit(computeType);
    const welfordCombine = generateWelfordCombine(computeType);
    const welfordFinalize = generateWelfordFinalize(computeType, config.applySqrt, outputType);

    const inputToCompute = generateCastSnippet('raw_val', getComputeType(inputDtype), computeType);

    return `
// ============================================================================
// Welford Global Reduction (Naive/Single-pass): ${dispatchKey}
// ============================================================================

${welfordStruct}

${welfordInit}

${welfordCombine}

${welfordFinalize}

struct Uniforms {
    numel: u32,
    correction: f32,
    inputOffset: u32,
    outputOffset: u32,
}

@group(0) @binding(0) var<storage, read> input: array<${getComputeType(inputDtype)}>;
@group(0) @binding(1) var<storage, read_write> output: array<${outputType}>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

var<workgroup> shared_mem: array<WelfordData, ${workgroupSize}>;

@compute @workgroup_size(${workgroupSize})
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let thread_id = local_id.x;
    let stride = ${workgroupSize}u;
    
    // Step 1: 每个线程累积多个元素
    var local_state = welford_empty();
    for (var i = thread_id; i < uniforms.numel; i += stride) {
        let raw_val = input[uniforms.inputOffset + i];
        let val = ${inputToCompute};
        local_state = welford_combine(local_state, welford_init(val));
    }
    
    // Step 2: 存入共享内存
    shared_mem[thread_id] = local_state;
    workgroupBarrier();
    
    // Step 3: Tree reduction
    for (var s = ${workgroupSize}u / 2u; s > 0u; s >>= 1u) {
        if (thread_id < s) {
            shared_mem[thread_id] = welford_combine(shared_mem[thread_id], shared_mem[thread_id + s]);
        }
        workgroupBarrier();
    }
    
    // Step 4: 写入结果
    if (thread_id == 0u) {
        let final_state = shared_mem[0];
        let correction = ${computeType}(uniforms.correction);
        let result = welford_finalize(final_state, correction);
        output[uniforms.outputOffset] = result;
    }
}
`;
}

/**
 * 构建 Strided Naive Global Welford Shader (非连续输入)
 */
export function buildStridedNaiveGlobalWelfordShader(
    iter: ITensorIterator,
    dispatchKey: string,
    workgroupSize: number,
    rank: number
): string {
    const config = WELFORD_OPS[dispatchKey];
    if (!config) {
        throw new Error(`Unknown Welford operation: ${dispatchKey}`);
    }

    const inputDtype = iter.input(0).dtype;
    const outputDtype = iter.output(0).dtype;
    const computeType = getComputeType(inputDtype);
    const outputType = getComputeType(outputDtype);

    const welfordStruct = generateWelfordStruct(computeType);
    const welfordInit = generateWelfordInit(computeType);
    const welfordCombine = generateWelfordCombine(computeType);
    const welfordFinalize = generateWelfordFinalize(computeType, config.applySqrt, outputType);

    const inputToCompute = generateCastSnippet('raw_val', getComputeType(inputDtype), computeType);

    return `
// ============================================================================
// Welford Global Reduction (Strided/Single-pass): ${dispatchKey}
// ============================================================================

${welfordStruct}

${welfordInit}

${welfordCombine}

${welfordFinalize}

struct Uniforms {
    numel: u32,
    correction: f32,
    rank: u32,
    inputOffset: u32,
    shape: vec4<u32>,
    shape2: vec4<u32>,
    strides: vec4<u32>,
    strides2: vec4<u32>,
    outputOffset: u32,
}

@group(0) @binding(0) var<storage, read> input: array<${getComputeType(inputDtype)}>;
@group(0) @binding(1) var<storage, read_write> output: array<${outputType}>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

var<workgroup> shared_mem: array<WelfordData, ${workgroupSize}>;

// 从逻辑索引计算物理偏移
fn compute_offset(logical_idx: u32) -> u32 {
    var offset = uniforms.inputOffset;
    var remaining = logical_idx;
    ${generateStridedOffsetCode(rank)}
    return offset;
}

@compute @workgroup_size(${workgroupSize})
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let thread_id = local_id.x;
    let stride = ${workgroupSize}u;
    
    // Step 1: 每个线程累积多个元素
    var local_state = welford_empty();
    for (var i = thread_id; i < uniforms.numel; i += stride) {
        let physical_idx = compute_offset(i);
        let raw_val = input[physical_idx];
        let val = ${inputToCompute};
        local_state = welford_combine(local_state, welford_init(val));
    }
    
    // Step 2: 存入共享内存
    shared_mem[thread_id] = local_state;
    workgroupBarrier();
    
    // Step 3: Tree reduction
    for (var s = ${workgroupSize}u / 2u; s > 0u; s >>= 1u) {
        if (thread_id < s) {
            shared_mem[thread_id] = welford_combine(shared_mem[thread_id], shared_mem[thread_id + s]);
        }
        workgroupBarrier();
    }
    
    // Step 4: 写入结果
    if (thread_id == 0u) {
        let final_state = shared_mem[0];
        let correction = ${computeType}(uniforms.correction);
        let result = welford_finalize(final_state, correction);
        output[uniforms.outputOffset] = result;
    }
}
`;
}

// ============================================================================
// Multi-pass Global Reduction Shaders
// ============================================================================

/**
 * 构建 Global Welford Stage 1 Shader
 * 每个 workgroup 输出一个 WelfordData (作为 vec3 存储)
 */
export function buildGlobalWelfordStage1Shader(
    iter: ITensorIterator,
    dispatchKey: string,
    workgroupSize: number
): string {
    const inputDtype = iter.input(0).dtype;
    const computeType = getComputeType(inputDtype);

    const welfordStruct = generateWelfordStruct(computeType);
    const welfordInit = generateWelfordInit(computeType);
    const welfordCombine = generateWelfordCombine(computeType);

    const inputToCompute = generateCastSnippet('raw_val', getComputeType(inputDtype), computeType);

    return `
// ============================================================================
// Welford Global Reduction Stage 1: ${dispatchKey}
// 每个 workgroup 输出一个部分 WelfordData
// ============================================================================

${welfordStruct}

${welfordInit}

${welfordCombine}

struct Uniforms {
    numel: u32,
    inputOffset: u32,
}

@group(0) @binding(0) var<storage, read> input: array<${getComputeType(inputDtype)}>;
@group(0) @binding(1) var<storage, read_write> partial_results: array<vec4<${computeType}>>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

var<workgroup> shared_mem: array<WelfordData, ${workgroupSize}>;

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
    
    // Step 1: Grid-stride loop
    var local_state = welford_empty();
    for (var i = global_idx; i < uniforms.numel; i += total_threads) {
        let raw_val = input[uniforms.inputOffset + i];
        let val = ${inputToCompute};
        local_state = welford_combine(local_state, welford_init(val));
    }
    
    // Step 2: 存入共享内存
    shared_mem[thread_id] = local_state;
    workgroupBarrier();
    
    // Step 3: Tree reduction
    for (var s = ${workgroupSize}u / 2u; s > 0u; s >>= 1u) {
        if (thread_id < s) {
            shared_mem[thread_id] = welford_combine(shared_mem[thread_id], shared_mem[thread_id + s]);
        }
        workgroupBarrier();
    }
    
    // Step 4: 写入部分结果 (以 vec4 格式，第 4 分量未使用)
    if (thread_id == 0u) {
        let state = shared_mem[0];
        partial_results[workgroup_idx] = vec4<${computeType}>(state.mean, state.m2, state.n, ${computeType}(0));
    }
}
`;
}

/**
 * 构建 Global Welford Stage 2 Shader
 * 合并所有部分结果并输出最终结果
 */
export function buildGlobalWelfordStage2Shader(
    dispatchKey: string,
    computeType: WgslDType,
    outputType: WgslDType,
    workgroupSize: number
): string {
    const config = WELFORD_OPS[dispatchKey];
    if (!config) {
        throw new Error(`Unknown Welford operation: ${dispatchKey}`);
    }

    const welfordStruct = generateWelfordStruct(computeType);
    const welfordCombine = generateWelfordCombine(computeType);
    const welfordFinalize = generateWelfordFinalize(computeType, config.applySqrt, outputType);

    return `
// ============================================================================
// Welford Global Reduction Stage 2: ${dispatchKey}
// 合并部分结果并计算最终 variance/std
// ============================================================================

${welfordStruct}

fn welford_empty() -> WelfordData {
    return WelfordData(${computeType}(0), ${computeType}(0), ${computeType}(0));
}

${welfordCombine}

${welfordFinalize}

struct Uniforms {
    numPartials: u32,
    correction: f32,
    outputOffset: u32,
}

@group(0) @binding(0) var<storage, read> partial_results: array<vec4<${computeType}>>;
@group(0) @binding(1) var<storage, read_write> output: array<${outputType}>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

var<workgroup> shared_mem: array<WelfordData, ${workgroupSize}>;

@compute @workgroup_size(${workgroupSize})
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let thread_id = local_id.x;
    let stride = ${workgroupSize}u;
    
    // Step 1: 加载部分结果
    var local_state = welford_empty();
    for (var i = thread_id; i < uniforms.numPartials; i += stride) {
        let packed = partial_results[i];
        let state = WelfordData(packed.x, packed.y, packed.z);
        local_state = welford_combine(local_state, state);
    }
    
    // Step 2: 存入共享内存
    shared_mem[thread_id] = local_state;
    workgroupBarrier();
    
    // Step 3: Tree reduction
    for (var s = ${workgroupSize}u / 2u; s > 0u; s >>= 1u) {
        if (thread_id < s) {
            shared_mem[thread_id] = welford_combine(shared_mem[thread_id], shared_mem[thread_id + s]);
        }
        workgroupBarrier();
    }
    
    // Step 4: 写入最终结果
    if (thread_id == 0u) {
        let final_state = shared_mem[0];
        let correction = ${computeType}(uniforms.correction);
        let result = welford_finalize(final_state, correction);
        output[uniforms.outputOffset] = result;
    }
}
`;
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * 生成 parallel offset 计算代码 (用于 dimensional reduction)
 */
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

/**
 * 生成 reduction offset 计算代码
 */
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

/**
 * 生成 strided offset 代码 (用于非连续全局归约)
 */
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
