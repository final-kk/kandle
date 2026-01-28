/**
 * LogSumExp Shader Builder
 * 
 * 生成 logsumexp 的 WGSL shader
 * 
 * 核心算法 (Two-Pass):
 * 1. 计算 max(x) - 数值稳定性的关键
 * 2. 计算 max + log(sum(exp(x - max)))
 * 
 * 实现策略:
 * - 使用 (max, sumExp) 二元组作为中间状态
 * - 支持 parallel combine: 合并两个部分结果
 * - 单 pass shader: 同时计算 max 和 sumExp (需要两次遍历共享内存)
 * 
 * 参考: PyTorch ATen/native/ReduceOps.cpp logsumexp_impl
 */

import type { ITensorIterator } from '@kandle/types';
import { getComputeType, generateCastSnippet } from '../../base/dtype';
import type { WgslDType } from '../../types';
import { Logger } from '@kandle/utils';

const logger = new Logger('LogSumExp-ShaderBuilder');

// ============================================================================
// Core WGSL Snippets
// ============================================================================

/**
 * 生成 LogSumExpState 结构体定义
 */
function generateLogSumExpStruct(computeType: string): string {
    return `
// LogSumExp 二元组: (max, sumExp)
// max: 当前已见元素的最大值
// sumExp: sum(exp(x - max_val)) 的累积和
struct LogSumExpState {
    max_val: ${computeType},    // 当前最大值
    sum_exp: ${computeType},    // sum(exp(x - max_val))
}
`;
}

/**
 * 生成 init 函数
 */
function generateLogSumExpInit(computeType: string): string {
    return `
// 初始化: 将单个值转换为 LogSumExpState
fn lse_init(val: ${computeType}) -> LogSumExpState {
    return LogSumExpState(val, ${computeType}(1));  // exp(val - val) = 1
}

// 初始化: 空状态 (用于越界线程)
fn lse_empty() -> LogSumExpState {
    return LogSumExpState(${computeType}(-1e38), ${computeType}(0));  // max=-inf, sum=0
}
`;
}

/**
 * 生成 combine 函数
 * 
 * 合并两个 LogSumExpState:
 * - 新 max = max(a.max, b.max)
 * - 新 sumExp = a.sumExp * exp(a.max - new_max) + b.sumExp * exp(b.max - new_max)
 */
function generateLogSumExpCombine(computeType: string): string {
    return `
// 合并两个 LogSumExpState
// 数学原理:
// log(sum(exp(A)) + sum(exp(B))) 
// = log(exp(max_A) * sum(exp(A - max_A)) + exp(max_B) * sum(exp(B - max_B)))
// = new_max + log(sumExp_A * exp(max_A - new_max) + sumExp_B * exp(max_B - new_max))
fn lse_combine(a: LogSumExpState, b: LogSumExpState) -> LogSumExpState {
    // 处理空状态
    if (a.sum_exp == ${computeType}(0)) { return b; }
    if (b.sum_exp == ${computeType}(0)) { return a; }
    
    // 新的最大值
    let new_max = max(a.max_val, b.max_val);
    
    // 重新缩放 sumExp
    // exp(old_max - new_max) 将旧的 sumExp 调整到新的尺度
    let sum_exp_new = a.sum_exp * exp(a.max_val - new_max) 
                    + b.sum_exp * exp(b.max_val - new_max);
    
    return LogSumExpState(new_max, sum_exp_new);
}
`;
}

/**
 * 生成 finalize 函数
 */
function generateLogSumExpFinalize(computeType: WgslDType, outputType: WgslDType): string {
    const castToOutput = computeType !== outputType
        ? generateCastSnippet('result', computeType, outputType)
        : 'result';

    return `
// 计算最终结果: max + log(sumExp)
fn lse_finalize(state: LogSumExpState) -> ${outputType} {
    // 处理空输入
    if (state.sum_exp <= ${computeType}(0)) {
        return ${outputType}(-1e38);  // -inf
    }
    
    let result = state.max_val + log(state.sum_exp);
    return ${castToOutput};
}
`;
}

// ============================================================================
// Dimensional Reduction Shader
// ============================================================================

/**
 * 构建 Dimensional LogSumExp Reduction Shader
 * 
 * 每个 workgroup 处理一个输出元素，归约其对应的所有输入元素
 */
export function buildDimLogSumExpShader(
    iter: ITensorIterator,
    workgroupSize: number
): string {
    const inputDtype = iter.input(0).dtype;
    const outputDtype = iter.output(0).dtype;
    const computeType = getComputeType(inputDtype);
    const outputType = getComputeType(outputDtype);

    const rank = iter.inputShape.length;
    const reductionRank = iter.reductionShape.length;

    logger.debug(`Building dim LogSumExp shader: rank=${rank}, reductionRank=${reductionRank}`);

    // 生成所有函数
    const lseStruct = generateLogSumExpStruct(computeType);
    const lseInit = generateLogSumExpInit(computeType);
    const lseCombine = generateLogSumExpCombine(computeType);
    const lseFinalize = generateLogSumExpFinalize(computeType, outputType);

    // Cast snippets
    const inputToCompute = generateCastSnippet('raw_val', getComputeType(inputDtype), computeType);

    return `
// ============================================================================
// LogSumExp Dimensional Reduction
// ============================================================================

${lseStruct}

${lseInit}

${lseCombine}

${lseFinalize}

// Uniform buffer layout (与 Welford 相同)
struct Uniforms {
    outputNumel: u32,           // 输出元素数
    reductionNumel: u32,        // 每个输出要归约的元素数
    padding0: u32,
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

var<workgroup> shared_mem: array<LogSumExpState, ${workgroupSize}>;

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
    var local_state = lse_empty();
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
        local_state = lse_combine(local_state, lse_init(val));
    }
    
    // Step 3: 存入共享内存
    shared_mem[thread_id] = local_state;
    workgroupBarrier();
    
    // Step 4: Tree reduction in shared memory
    for (var s = ${workgroupSize}u / 2u; s > 0u; s >>= 1u) {
        if (thread_id < s) {
            shared_mem[thread_id] = lse_combine(shared_mem[thread_id], shared_mem[thread_id + s]);
        }
        workgroupBarrier();
    }
    
    // Step 5: 线程 0 写入最终结果
    if (thread_id == 0u) {
        let final_state = shared_mem[0];
        let result = lse_finalize(final_state);
        output[uniforms.outputOffset + output_idx] = result;
    }
}
`;
}

// ============================================================================
// Global Reduction Shaders
// ============================================================================

/**
 * 构建 Naive Global LogSumExp Shader (单 workgroup)
 */
export function buildNaiveGlobalLogSumExpShader(
    iter: ITensorIterator,
    workgroupSize: number
): string {
    const inputDtype = iter.input(0).dtype;
    const outputDtype = iter.output(0).dtype;
    const computeType = getComputeType(inputDtype);
    const outputType = getComputeType(outputDtype);

    const lseStruct = generateLogSumExpStruct(computeType);
    const lseInit = generateLogSumExpInit(computeType);
    const lseCombine = generateLogSumExpCombine(computeType);
    const lseFinalize = generateLogSumExpFinalize(computeType, outputType);

    const inputToCompute = generateCastSnippet('raw_val', getComputeType(inputDtype), computeType);

    return `
// ============================================================================
// LogSumExp Global Reduction (Naive/Single-pass)
// ============================================================================

${lseStruct}

${lseInit}

${lseCombine}

${lseFinalize}

struct Uniforms {
    numel: u32,
    inputOffset: u32,
    outputOffset: u32,
}

@group(0) @binding(0) var<storage, read> input: array<${getComputeType(inputDtype)}>;
@group(0) @binding(1) var<storage, read_write> output: array<${outputType}>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

var<workgroup> shared_mem: array<LogSumExpState, ${workgroupSize}>;

@compute @workgroup_size(${workgroupSize})
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let thread_id = local_id.x;
    let stride = ${workgroupSize}u;
    
    // Step 1: 每个线程累积多个元素
    var local_state = lse_empty();
    for (var i = thread_id; i < uniforms.numel; i += stride) {
        let raw_val = input[uniforms.inputOffset + i];
        let val = ${inputToCompute};
        local_state = lse_combine(local_state, lse_init(val));
    }
    
    // Step 2: 存入共享内存
    shared_mem[thread_id] = local_state;
    workgroupBarrier();
    
    // Step 3: Tree reduction
    for (var s = ${workgroupSize}u / 2u; s > 0u; s >>= 1u) {
        if (thread_id < s) {
            shared_mem[thread_id] = lse_combine(shared_mem[thread_id], shared_mem[thread_id + s]);
        }
        workgroupBarrier();
    }
    
    // Step 4: 写入结果
    if (thread_id == 0u) {
        let final_state = shared_mem[0];
        let result = lse_finalize(final_state);
        output[uniforms.outputOffset] = result;
    }
}
`;
}

/**
 * 构建 Strided Naive Global LogSumExp Shader (非连续输入)
 */
export function buildStridedNaiveGlobalLogSumExpShader(
    iter: ITensorIterator,
    workgroupSize: number,
    rank: number
): string {
    const inputDtype = iter.input(0).dtype;
    const outputDtype = iter.output(0).dtype;
    const computeType = getComputeType(inputDtype);
    const outputType = getComputeType(outputDtype);

    const lseStruct = generateLogSumExpStruct(computeType);
    const lseInit = generateLogSumExpInit(computeType);
    const lseCombine = generateLogSumExpCombine(computeType);
    const lseFinalize = generateLogSumExpFinalize(computeType, outputType);

    const inputToCompute = generateCastSnippet('raw_val', getComputeType(inputDtype), computeType);

    return `
// ============================================================================
// LogSumExp Global Reduction (Strided/Single-pass)
// ============================================================================

${lseStruct}

${lseInit}

${lseCombine}

${lseFinalize}

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

var<workgroup> shared_mem: array<LogSumExpState, ${workgroupSize}>;

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
    var local_state = lse_empty();
    for (var i = thread_id; i < uniforms.numel; i += stride) {
        let physical_idx = compute_offset(i);
        let raw_val = input[physical_idx];
        let val = ${inputToCompute};
        local_state = lse_combine(local_state, lse_init(val));
    }
    
    // Step 2: 存入共享内存
    shared_mem[thread_id] = local_state;
    workgroupBarrier();
    
    // Step 3: Tree reduction
    for (var s = ${workgroupSize}u / 2u; s > 0u; s >>= 1u) {
        if (thread_id < s) {
            shared_mem[thread_id] = lse_combine(shared_mem[thread_id], shared_mem[thread_id + s]);
        }
        workgroupBarrier();
    }
    
    // Step 4: 写入结果
    if (thread_id == 0u) {
        let final_state = shared_mem[0];
        let result = lse_finalize(final_state);
        output[uniforms.outputOffset] = result;
    }
}
`;
}

/**
 * 构建 Global LogSumExp Stage 1 Shader
 * 每个 workgroup 输出一个 LogSumExpState (作为 vec2 存储)
 */
export function buildGlobalLogSumExpStage1Shader(
    iter: ITensorIterator,
    workgroupSize: number
): string {
    const inputDtype = iter.input(0).dtype;
    const computeType = getComputeType(inputDtype);

    const lseStruct = generateLogSumExpStruct(computeType);
    const lseInit = generateLogSumExpInit(computeType);
    const lseCombine = generateLogSumExpCombine(computeType);

    const inputToCompute = generateCastSnippet('raw_val', getComputeType(inputDtype), computeType);

    return `
// ============================================================================
// LogSumExp Global Reduction Stage 1
// 每个 workgroup 输出一个部分 LogSumExpState
// ============================================================================

${lseStruct}

${lseInit}

${lseCombine}

struct Uniforms {
    numel: u32,
    inputOffset: u32,
}

@group(0) @binding(0) var<storage, read> input: array<${getComputeType(inputDtype)}>;
@group(0) @binding(1) var<storage, read_write> partial_results: array<vec2<${computeType}>>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

var<workgroup> shared_mem: array<LogSumExpState, ${workgroupSize}>;

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
    var local_state = lse_empty();
    for (var i = global_idx; i < uniforms.numel; i += total_threads) {
        let raw_val = input[uniforms.inputOffset + i];
        let val = ${inputToCompute};
        local_state = lse_combine(local_state, lse_init(val));
    }
    
    // Step 2: 存入共享内存
    shared_mem[thread_id] = local_state;
    workgroupBarrier();
    
    // Step 3: Tree reduction
    for (var s = ${workgroupSize}u / 2u; s > 0u; s >>= 1u) {
        if (thread_id < s) {
            shared_mem[thread_id] = lse_combine(shared_mem[thread_id], shared_mem[thread_id + s]);
        }
        workgroupBarrier();
    }
    
    // Step 4: 写入部分结果 (以 vec2 格式)
    if (thread_id == 0u) {
        let state = shared_mem[0];
        partial_results[workgroup_idx] = vec2<${computeType}>(state.max_val, state.sum_exp);
    }
}
`;
}

/**
 * 构建 Global LogSumExp Stage 2 Shader
 * 合并所有部分结果并输出最终结果
 */
export function buildGlobalLogSumExpStage2Shader(
    computeType: WgslDType,
    outputType: WgslDType,
    workgroupSize: number
): string {
    const lseStruct = generateLogSumExpStruct(computeType);
    const lseCombine = generateLogSumExpCombine(computeType);
    const lseFinalize = generateLogSumExpFinalize(computeType, outputType);

    return `
// ============================================================================
// LogSumExp Global Reduction Stage 2
// 合并部分结果并计算最终 logsumexp
// ============================================================================

${lseStruct}

fn lse_empty() -> LogSumExpState {
    return LogSumExpState(${computeType}(-1e38), ${computeType}(0));
}

${lseCombine}

${lseFinalize}

struct Uniforms {
    numPartials: u32,
    outputOffset: u32,
}

@group(0) @binding(0) var<storage, read> partial_results: array<vec2<${computeType}>>;
@group(0) @binding(1) var<storage, read_write> output: array<${outputType}>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

var<workgroup> shared_mem: array<LogSumExpState, ${workgroupSize}>;

@compute @workgroup_size(${workgroupSize})
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let thread_id = local_id.x;
    let stride = ${workgroupSize}u;
    
    // Step 1: 加载部分结果
    var local_state = lse_empty();
    for (var i = thread_id; i < uniforms.numPartials; i += stride) {
        let packed = partial_results[i];
        let state = LogSumExpState(packed.x, packed.y);
        local_state = lse_combine(local_state, state);
    }
    
    // Step 2: 存入共享内存
    shared_mem[thread_id] = local_state;
    workgroupBarrier();
    
    // Step 3: Tree reduction
    for (var s = ${workgroupSize}u / 2u; s > 0u; s >>= 1u) {
        if (thread_id < s) {
            shared_mem[thread_id] = lse_combine(shared_mem[thread_id], shared_mem[thread_id + s]);
        }
        workgroupBarrier();
    }
    
    // Step 4: 写入最终结果
    if (thread_id == 0u) {
        let final_state = shared_mem[0];
        let result = lse_finalize(final_state);
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
