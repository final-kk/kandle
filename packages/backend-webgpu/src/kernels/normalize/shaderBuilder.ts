/**
 * Normalize Shader Builder
 * 
 * 生成 Softmax/LayerNorm/BatchNorm 等归一化操作的 WGSL Shader
 * 
 * 设计原则:
 * 1. 单 pass fused kernel，避免多次内存访问
 * 2. 数值稳定性: softmax 先减 max，variance 用 Welford
 * 3. 支持任意维度归一化
 * 
 * 工业级实现：原生支持 strided (非连续) 输入
 * - 输入使用 shape/strides/offset 计算物理地址
 * - 输出总是连续的，使用 flat index
 */

import type { NormalizeOpConfig, NormalizeShaderParams } from './types';
import { getComputeType } from '../../base/dtype';
import type { DType } from '@kandle/types';

/**
 * 构建 Normalize Shader
 */
export function buildNormalizeShader(params: NormalizeShaderParams): string {
    const {
        config,
        inputShape,
        inputStrides,
        inputOffset,
        normalizedDims,
        reduceSize,
        hasWeight,
        hasBias,
        eps,
        p,
        dtype,
    } = params;

    const wgslType = getComputeType(dtype as DType);
    const workgroupSize = 256;

    switch (config.kind) {
        case 'softmax':
        case 'log_softmax':
        case 'softmin':
            return buildSoftmaxShader({
                inputShape,
                inputStrides,
                inputOffset,
                normalizedDims,
                reduceSize,
                wgslType,
                workgroupSize,
                isLogSoftmax: config.kind === 'log_softmax',
                negateInput: config.negateInput ?? false,
            });

        case 'layer_norm':
            return buildLayerNormShader({
                inputShape,
                inputStrides,
                inputOffset,
                normalizedDims,
                reduceSize,
                hasWeight,
                hasBias,
                eps,
                wgslType,
                workgroupSize,
            });

        case 'rms_norm':
            return buildRMSNormShader({
                inputShape,
                inputStrides,
                inputOffset,
                normalizedDims,
                reduceSize,
                hasWeight,
                eps,
                wgslType,
                workgroupSize,
            });

        case 'batch_norm':
            return buildBatchNormShader({
                inputShape,
                inputStrides,
                inputOffset,
                hasWeight,
                hasBias,
                eps,
                wgslType,
                workgroupSize,
            });

        case 'group_norm':
            return buildGroupNormShader({
                inputShape,
                inputStrides,
                inputOffset,
                numGroups: params.numGroups ?? 1,
                hasWeight,
                hasBias,
                eps,
                wgslType,
                workgroupSize,
            });

        case 'lp_normalize':
            return buildLpNormalizeShader({
                inputShape,
                inputStrides,
                inputOffset,
                normalizedDims,
                reduceSize,
                p: p ?? 2,
                eps,
                wgslType,
                workgroupSize,
            });

        default:
            throw new Error(`Unknown normalize kind: ${config.kind}`);
    }
}

// ============================================================================
// Softmax Shader (Strided Input, Contiguous Output)
// ============================================================================

interface SoftmaxShaderParams {
    inputShape: readonly number[];
    inputStrides: readonly number[];
    inputOffset: number;
    normalizedDims: number[];
    reduceSize: number;
    wgslType: string;
    workgroupSize: number;
    isLogSoftmax: boolean;
    negateInput: boolean;
}

function buildSoftmaxShader(params: SoftmaxShaderParams): string {
    const {
        inputShape,
        inputStrides,
        inputOffset,
        normalizedDims,
        reduceSize,
        wgslType,
        workgroupSize,
        isLogSoftmax,
        negateInput,
    } = params;

    const ndim = inputShape.length;
    const totalSize = inputShape.reduce((a, b) => a * b, 1);
    const batchSize = totalSize / reduceSize;

    // 生成 strided 索引计算代码
    const indexingCode = generateStridedIndexingCode(inputShape, inputStrides, inputOffset, normalizedDims);

    const inputTransform = negateInput ? '-' : '';
    const outputExpr = isLogSoftmax
        ? `(val - max_val - log(sum_exp))`
        : `(exp(val - max_val) / sum_exp)`;

    return `
// Softmax/LogSoftmax Shader (Strided Input, Contiguous Output)
// Shape: [${inputShape.join(', ')}]
// Strides: [${inputStrides.join(', ')}]
// Reduce dims: [${normalizedDims.join(', ')}]
// Reduce size: ${reduceSize}

@group(0) @binding(0) var<storage, read> input: array<${wgslType}>;
@group(0) @binding(1) var<storage, read_write> output: array<${wgslType}>;

${indexingCode.constants}
const REDUCE_SIZE: u32 = ${reduceSize}u;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.x;
    if (batch_idx >= ${batchSize}u) {
        return;
    }

    // 计算 batch 的起始偏移 (输入用 strides，输出用 flat)
    ${indexingCode.computeBatchOffset}
    let output_batch_offset = batch_idx * REDUCE_SIZE;

    // Step 1: 找 max (数值稳定性)
    var max_val: ${wgslType} = ${wgslType}(-1e38);
    for (var i = 0u; i < REDUCE_SIZE; i++) {
        ${indexingCode.computeElementOffset}
        let val = ${inputTransform}input[input_offset];
        max_val = max(max_val, val);
    }

    // Step 2: 计算 sum(exp(x - max))
    var sum_exp: ${wgslType} = ${wgslType}(0);
    for (var i = 0u; i < REDUCE_SIZE; i++) {
        ${indexingCode.computeElementOffset}
        let val = ${inputTransform}input[input_offset];
        sum_exp += exp(val - max_val);
    }

    // Step 3: 归一化输出 (写入连续输出)
    for (var i = 0u; i < REDUCE_SIZE; i++) {
        ${indexingCode.computeElementOffset}
        let val = ${inputTransform}input[input_offset];
        let output_offset = output_batch_offset + i;
        output[output_offset] = ${wgslType}${outputExpr};
    }
}
`;
}

// ============================================================================
// Layer Norm Shader (Strided Input, Contiguous Output)
// ============================================================================

interface LayerNormShaderParams {
    inputShape: readonly number[];
    inputStrides: readonly number[];
    inputOffset: number;
    normalizedDims: number[];
    reduceSize: number;
    hasWeight: boolean;
    hasBias: boolean;
    eps: number;
    wgslType: string;
    workgroupSize: number;
}

function buildLayerNormShader(params: LayerNormShaderParams): string {
    const {
        inputShape,
        inputStrides,
        inputOffset,
        normalizedDims,
        reduceSize,
        hasWeight,
        hasBias,
        eps,
        wgslType,
        workgroupSize,
    } = params;

    const totalSize = inputShape.reduce((a, b) => a * b, 1);
    const batchSize = totalSize / reduceSize;
    const indexingCode = generateStridedIndexingCode(inputShape, inputStrides, inputOffset, normalizedDims);

    // Binding 声明
    let bindingIdx = 2;
    const weightBinding = hasWeight ? `@group(0) @binding(${bindingIdx++}) var<storage, read> weight: array<${wgslType}>;` : '';
    const biasBinding = hasBias ? `@group(0) @binding(${bindingIdx++}) var<storage, read> bias: array<${wgslType}>;` : '';

    // 输出表达式
    let outputExpr = `normalized`;
    if (hasWeight) {
        outputExpr = `${outputExpr} * weight[i]`;
    }
    if (hasBias) {
        outputExpr = `${outputExpr} + bias[i]`;
    }

    return `
// Layer Normalization Shader (Strided Input, Contiguous Output, Welford Algorithm)
// Shape: [${inputShape.join(', ')}]
// Strides: [${inputStrides.join(', ')}]
// Normalized dims: [${normalizedDims.join(', ')}]
// Reduce size: ${reduceSize}

@group(0) @binding(0) var<storage, read> input: array<${wgslType}>;
@group(0) @binding(1) var<storage, read_write> output: array<${wgslType}>;
${weightBinding}
${biasBinding}

${indexingCode.constants}
const REDUCE_SIZE: u32 = ${reduceSize}u;

const eps: ${wgslType} = ${wgslType}(${eps});
const reduce_size_f: ${wgslType} = ${wgslType}(${reduceSize});

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.x;
    if (batch_idx >= ${batchSize}u) {
        return;
    }

    // 计算 batch 的起始偏移
    ${indexingCode.computeBatchOffset}
    let output_batch_offset = batch_idx * REDUCE_SIZE;

    // Welford 在线算法计算 mean 和 variance
    var mean: ${wgslType} = ${wgslType}(0);
    var m2: ${wgslType} = ${wgslType}(0);
    var count: ${wgslType} = ${wgslType}(0);

    for (var i = 0u; i < REDUCE_SIZE; i++) {
        ${indexingCode.computeElementOffset}
        let val = input[input_offset];
        count += ${wgslType}(1);
        let delta = val - mean;
        mean += delta / count;
        let delta2 = val - mean;
        m2 += delta * delta2;
    }

    let variance = m2 / reduce_size_f;
    let rstd = ${wgslType}(1) / sqrt(variance + eps);

    // 归一化并应用仿射变换 (写入连续输出)
    for (var i = 0u; i < REDUCE_SIZE; i++) {
        ${indexingCode.computeElementOffset}
        let val = input[input_offset];
        let normalized = (val - mean) * rstd;
        let output_offset = output_batch_offset + i;
        output[output_offset] = ${wgslType}(${outputExpr});
    }
}
`;
}

// ============================================================================
// RMS Norm Shader (Strided Input, Contiguous Output)
// ============================================================================

interface RMSNormShaderParams {
    inputShape: readonly number[];
    inputStrides: readonly number[];
    inputOffset: number;
    normalizedDims: number[];
    reduceSize: number;
    hasWeight: boolean;
    eps: number;
    wgslType: string;
    workgroupSize: number;
}

function buildRMSNormShader(params: RMSNormShaderParams): string {
    const {
        inputShape,
        inputStrides,
        inputOffset,
        normalizedDims,
        reduceSize,
        hasWeight,
        eps,
        wgslType,
        workgroupSize,
    } = params;

    const totalSize = inputShape.reduce((a, b) => a * b, 1);
    const batchSize = totalSize / reduceSize;
    const indexingCode = generateStridedIndexingCode(inputShape, inputStrides, inputOffset, normalizedDims);

    const weightBinding = hasWeight
        ? `@group(0) @binding(2) var<storage, read> weight: array<${wgslType}>;`
        : '';

    const outputExpr = hasWeight ? `normalized * weight[i]` : `normalized`;

    return `
// RMS Normalization Shader (Strided Input, Contiguous Output)
// Shape: [${inputShape.join(', ')}]
// Strides: [${inputStrides.join(', ')}]
// Normalized dims: [${normalizedDims.join(', ')}]
// Reduce size: ${reduceSize}

@group(0) @binding(0) var<storage, read> input: array<${wgslType}>;
@group(0) @binding(1) var<storage, read_write> output: array<${wgslType}>;
${weightBinding}

${indexingCode.constants}
const REDUCE_SIZE: u32 = ${reduceSize}u;

const eps: ${wgslType} = ${wgslType}(${eps});
const reduce_size_f: ${wgslType} = ${wgslType}(${reduceSize});

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.x;
    if (batch_idx >= ${batchSize}u) {
        return;
    }

    // 计算 batch 的起始偏移
    ${indexingCode.computeBatchOffset}
    let output_batch_offset = batch_idx * REDUCE_SIZE;

    // 计算 mean(x²)
    var sum_sq: ${wgslType} = ${wgslType}(0);
    for (var i = 0u; i < REDUCE_SIZE; i++) {
        ${indexingCode.computeElementOffset}
        let val = input[input_offset];
        sum_sq += val * val;
    }

    let rms = sqrt(sum_sq / reduce_size_f + eps);
    let scale = ${wgslType}(1) / rms;

    // 归一化 (写入连续输出)
    for (var i = 0u; i < REDUCE_SIZE; i++) {
        ${indexingCode.computeElementOffset}
        let val = input[input_offset];
        let normalized = val * scale;
        let output_offset = output_batch_offset + i;
        output[output_offset] = ${wgslType}(${outputExpr});
    }
}
`;
}

// ============================================================================
// Batch Norm Shader (Strided Input, Contiguous Output)
// ============================================================================

interface BatchNormShaderParams {
    inputShape: readonly number[];
    inputStrides: readonly number[];
    inputOffset: number;
    hasWeight: boolean;
    hasBias: boolean;
    eps: number;
    wgslType: string;
    workgroupSize: number;
}

function buildBatchNormShader(params: BatchNormShaderParams): string {
    const {
        inputShape,
        inputStrides,
        inputOffset,
        hasWeight,
        hasBias,
        eps,
        wgslType,
        workgroupSize,
    } = params;

    // BatchNorm 对 NCHW 数据: 沿 N, H, W 归一化，每个 C 独立
    // 输入 shape: [N, C, H, W] 或 [N, C, *]
    const N = inputShape[0];
    const C = inputShape[1];
    const spatialSize = inputShape.slice(2).reduce((a, b) => a * b, 1);
    const totalSize = N * C * spatialSize;

    const ndim = inputShape.length;

    let bindingIdx = 2;
    const runningMeanBinding = `@group(0) @binding(${bindingIdx++}) var<storage, read> running_mean: array<${wgslType}>;`;
    const runningVarBinding = `@group(0) @binding(${bindingIdx++}) var<storage, read> running_var: array<${wgslType}>;`;
    const weightBinding = hasWeight
        ? `@group(0) @binding(${bindingIdx++}) var<storage, read> weight: array<${wgslType}>;`
        : '';
    const biasBinding = hasBias
        ? `@group(0) @binding(${bindingIdx++}) var<storage, read> bias: array<${wgslType}>;`
        : '';

    let outputExpr = `normalized`;
    if (hasWeight) outputExpr = `${outputExpr} * weight[c]`;
    if (hasBias) outputExpr = `${outputExpr} + bias[c]`;

    // 生成 strides 常量
    const stridesConst = `const INPUT_STRIDES: array<i32, ${ndim}> = array<i32, ${ndim}>(${inputStrides.join(', ')});`;

    return `
// Batch Normalization Shader (Strided Input, Contiguous Output, Inference Mode)
// Shape: [${inputShape.join(', ')}]
// Strides: [${inputStrides.join(', ')}]
// N=${N}, C=${C}, spatial=${spatialSize}

@group(0) @binding(0) var<storage, read> input: array<${wgslType}>;
@group(0) @binding(1) var<storage, read_write> output: array<${wgslType}>;
${runningMeanBinding}
${runningVarBinding}
${weightBinding}
${biasBinding}

${stridesConst}
const INPUT_OFFSET: u32 = ${inputOffset}u;
const eps: ${wgslType} = ${wgslType}(${eps});

// 计算 strided 物理偏移 (输入)
fn compute_input_offset(n: u32, c: u32, s: u32) -> u32 {
    var offset: i32 = i32(INPUT_OFFSET);
    // n 维度
    offset += i32(n) * INPUT_STRIDES[0];
    // c 维度
    offset += i32(c) * INPUT_STRIDES[1];
    // 空间维度 (展平处理)
    var rem = s;
    ${inputShape.slice(2).map((_, i) => {
        const dimIdx = i + 2;
        const suffix = inputShape.slice(dimIdx + 1).reduce((a, b) => a * b, 1);
        return `{
        let coord_${dimIdx} = rem / ${suffix}u;
        rem = rem % ${suffix}u;
        offset += i32(coord_${dimIdx}) * INPUT_STRIDES[${dimIdx}];
    }`;
    }).join('\n    ')}
    return u32(offset);
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= ${totalSize}u) {
        return;
    }

    // 计算 n, c, spatial_idx (逻辑坐标)
    let c = (idx / ${spatialSize}u) % ${C}u;
    let n = idx / ${C * spatialSize}u;
    let s = idx % ${spatialSize}u;

    // 计算输入物理偏移 (strided)
    let input_physical_offset = compute_input_offset(n, c, s);
    // 输出偏移 = 连续 idx
    let output_offset = idx;

    let mean = running_mean[c];
    let variance = running_var[c];
    let rstd = ${wgslType}(1) / sqrt(variance + eps);

    let val = input[input_physical_offset];
    let normalized = (val - mean) * rstd;
    output[output_offset] = ${wgslType}(${outputExpr});
}
`;
}

// ============================================================================
// Group Norm Shader (Strided Input, Contiguous Output)
// ============================================================================

interface GroupNormShaderParams {
    inputShape: readonly number[];
    inputStrides: readonly number[];
    inputOffset: number;
    numGroups: number;
    hasWeight: boolean;
    hasBias: boolean;
    eps: number;
    wgslType: string;
    workgroupSize: number;
}

function buildGroupNormShader(params: GroupNormShaderParams): string {
    const {
        inputShape,
        inputStrides,
        inputOffset,
        numGroups,
        hasWeight,
        hasBias,
        eps,
        wgslType,
        workgroupSize,
    } = params;

    const N = inputShape[0];
    const C = inputShape[1];
    const spatialSize = inputShape.slice(2).reduce((a, b) => a * b, 1);
    const channelsPerGroup = Math.floor(C / numGroups);
    const groupSize = channelsPerGroup * spatialSize;

    const ndim = inputShape.length;

    let bindingIdx = 2;
    const weightBinding = hasWeight
        ? `@group(0) @binding(${bindingIdx++}) var<storage, read> weight: array<${wgslType}>;`
        : '';
    const biasBinding = hasBias
        ? `@group(0) @binding(${bindingIdx++}) var<storage, read> bias: array<${wgslType}>;`
        : '';

    let outputExpr = `normalized`;
    if (hasWeight) outputExpr = `${outputExpr} * weight[c]`;
    if (hasBias) outputExpr = `${outputExpr} + bias[c]`;

    // 生成 strides 常量
    const stridesConst = `const INPUT_STRIDES: array<i32, ${ndim}> = array<i32, ${ndim}>(${inputStrides.join(', ')});`;

    return `
// Group Normalization Shader (Strided Input, Contiguous Output)
// Shape: [${inputShape.join(', ')}]
// Strides: [${inputStrides.join(', ')}]
// N=${N}, C=${C}, spatial=${spatialSize}, numGroups=${numGroups}

@group(0) @binding(0) var<storage, read> input: array<${wgslType}>;
@group(0) @binding(1) var<storage, read_write> output: array<${wgslType}>;
${weightBinding}
${biasBinding}

${stridesConst}
const INPUT_OFFSET: u32 = ${inputOffset}u;
const eps: ${wgslType} = ${wgslType}(${eps});
const NUM_GROUPS: u32 = ${numGroups}u;
const CHANNELS_PER_GROUP: u32 = ${channelsPerGroup}u;
const GROUP_SIZE: u32 = ${groupSize}u;
const SPATIAL: u32 = ${spatialSize}u;

// 计算 strided 物理偏移 (输入)
fn compute_input_offset(n: u32, c: u32, s: u32) -> u32 {
    var offset: i32 = i32(INPUT_OFFSET);
    offset += i32(n) * INPUT_STRIDES[0];
    offset += i32(c) * INPUT_STRIDES[1];
    var rem = s;
    ${inputShape.slice(2).map((_, i) => {
        const dimIdx = i + 2;
        const suffix = inputShape.slice(dimIdx + 1).reduce((a, b) => a * b, 1);
        return `{
        let coord_${dimIdx} = rem / ${suffix}u;
        rem = rem % ${suffix}u;
        offset += i32(coord_${dimIdx}) * INPUT_STRIDES[${dimIdx}];
    }`;
    }).join('\n    ')}
    return u32(offset);
}

// 计算连续输出偏移
fn compute_output_offset(n: u32, c: u32, s: u32) -> u32 {
    return n * ${C * spatialSize}u + c * SPATIAL + s;
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // 每个 workgroup 处理一个 (n, g) 对
    let ng_idx = gid.x;
    if (ng_idx >= ${N * numGroups}u) {
        return;
    }

    let n = ng_idx / NUM_GROUPS;
    let g = ng_idx % NUM_GROUPS;

    // 计算 group 内的 mean 和 variance (Welford)
    var mean: ${wgslType} = ${wgslType}(0);
    var m2: ${wgslType} = ${wgslType}(0);
    var count: ${wgslType} = ${wgslType}(0);

    let base_c = g * CHANNELS_PER_GROUP;
    for (var c_offset = 0u; c_offset < CHANNELS_PER_GROUP; c_offset++) {
        let c = base_c + c_offset;
        for (var s = 0u; s < SPATIAL; s++) {
            let input_offset = compute_input_offset(n, c, s);
            let val = input[input_offset];
            count += ${wgslType}(1);
            let delta = val - mean;
            mean += delta / count;
            let delta2 = val - mean;
            m2 += delta * delta2;
        }
    }

    let variance = m2 / ${wgslType}(GROUP_SIZE);
    let rstd = ${wgslType}(1) / sqrt(variance + eps);

    // 归一化 (写入连续输出)
    for (var c_offset = 0u; c_offset < CHANNELS_PER_GROUP; c_offset++) {
        let c = base_c + c_offset;
        for (var s = 0u; s < SPATIAL; s++) {
            let input_offset = compute_input_offset(n, c, s);
            let output_offset = compute_output_offset(n, c, s);
            let val = input[input_offset];
            let normalized = (val - mean) * rstd;
            output[output_offset] = ${wgslType}(${outputExpr});
        }
    }
}
`;
}

// ============================================================================
// Lp Normalize Shader (Strided Input, Contiguous Output)
// ============================================================================

interface LpNormalizeShaderParams {
    inputShape: readonly number[];
    inputStrides: readonly number[];
    inputOffset: number;
    normalizedDims: number[];
    reduceSize: number;
    p: number;
    eps: number;
    wgslType: string;
    workgroupSize: number;
}

function buildLpNormalizeShader(params: LpNormalizeShaderParams): string {
    const {
        inputShape,
        inputStrides,
        inputOffset,
        normalizedDims,
        reduceSize,
        p,
        eps,
        wgslType,
        workgroupSize,
    } = params;

    const totalSize = inputShape.reduce((a, b) => a * b, 1);
    const batchSize = totalSize / reduceSize;
    const indexingCode = generateStridedIndexingCode(inputShape, inputStrides, inputOffset, normalizedDims);

    // 特殊处理 p=2 (L2 norm) 避免 pow 调用
    const normAccumExpr = p === 2
        ? `val * val`
        : `pow(abs(val), ${wgslType}(${p}))`;
    const normFinalExpr = p === 2
        ? `sqrt(norm_sum)`
        : `pow(norm_sum, ${wgslType}(${1 / p}))`;

    return `
// Lp Normalize Shader (Strided Input, Contiguous Output)
// Shape: [${inputShape.join(', ')}]
// Strides: [${inputStrides.join(', ')}]
// Normalized dims: [${normalizedDims.join(', ')}]
// p = ${p}

@group(0) @binding(0) var<storage, read> input: array<${wgslType}>;
@group(0) @binding(1) var<storage, read_write> output: array<${wgslType}>;

${indexingCode.constants}
const REDUCE_SIZE: u32 = ${reduceSize}u;

const eps: ${wgslType} = ${wgslType}(${eps});

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.x;
    if (batch_idx >= ${batchSize}u) {
        return;
    }

    // 计算 batch 的起始偏移
    ${indexingCode.computeBatchOffset}
    let output_batch_offset = batch_idx * REDUCE_SIZE;

    // 计算 Lp 范数
    var norm_sum: ${wgslType} = ${wgslType}(0);
    for (var i = 0u; i < REDUCE_SIZE; i++) {
        ${indexingCode.computeElementOffset}
        let val = input[input_offset];
        norm_sum += ${normAccumExpr};
    }

    let norm_val = max(${normFinalExpr}, eps);
    let scale = ${wgslType}(1) / norm_val;

    // 归一化 (写入连续输出)
    for (var i = 0u; i < REDUCE_SIZE; i++) {
        ${indexingCode.computeElementOffset}
        let val = input[input_offset];
        let output_offset = output_batch_offset + i;
        output[output_offset] = val * scale;
    }
}
`;
}

// ============================================================================
// 辅助函数：Strided 索引计算
// ============================================================================

interface StridedIndexingCode {
    constants: string;
    computeBatchOffset: string;
    computeElementOffset: string;
}

/**
 * 生成 strided 索引计算代码
 * 
 * 工业级实现：
 * - 输入使用 strides 计算物理地址
 * - 输出使用 flat index (batch_idx * reduceSize + i)
 */
function generateStridedIndexingCode(
    shape: readonly number[],
    strides: readonly number[],
    offset: number,
    reduceDims: number[]
): StridedIndexingCode {
    const ndim = shape.length;
    const batchDims: number[] = [];
    for (let d = 0; d < ndim; d++) {
        if (!reduceDims.includes(d)) {
            batchDims.push(d);
        }
    }

    // 生成 strides 常量 (使用 i32 支持负步幅)
    const stridesConst = `const INPUT_STRIDES: array<i32, ${ndim}> = array<i32, ${ndim}>(${strides.join(', ')});`;
    const shapeConst = `const INPUT_SHAPE: array<u32, ${ndim}> = array<u32, ${ndim}>(${shape.join(', ')});`;
    const offsetConst = `const INPUT_OFFSET: i32 = ${offset};`;

    // 计算 batch 维度的各维度的后缀积 (用于从 batch_idx 展开坐标)
    const batchSuffixes: number[] = batchDims.map((_, idx) =>
        batchDims.slice(idx + 1).map(d => shape[d]).reduce((a, b) => a * b, 1)
    );

    // 计算 reduce 维度的各维度的后缀积 (用于从 i 展开坐标)
    const reduceSuffixes: number[] = reduceDims.map((_, idx) =>
        reduceDims.slice(idx + 1).map(d => shape[d]).reduce((a, b) => a * b, 1)
    );

    // 生成 batch offset 计算代码 (输入的 strided 偏移)
    let computeBatchOffset = `var input_batch_offset: i32 = INPUT_OFFSET;\n    var batch_rem = batch_idx;\n`;
    for (let i = 0; i < batchDims.length; i++) {
        const d = batchDims[i];
        const suffix = batchSuffixes[i];
        computeBatchOffset += `    {\n`;
        computeBatchOffset += `        let coord = batch_rem / ${suffix}u;\n`;
        computeBatchOffset += `        batch_rem = batch_rem % ${suffix}u;\n`;
        computeBatchOffset += `        input_batch_offset += i32(coord) * INPUT_STRIDES[${d}];\n`;
        computeBatchOffset += `    }\n`;
    }

    // 生成 element offset 计算代码 (输入的 strided 偏移)
    let computeElementOffset = `var input_elem_offset: i32 = input_batch_offset;\n        var elem_rem = i;\n`;
    for (let i = 0; i < reduceDims.length; i++) {
        const d = reduceDims[i];
        const suffix = reduceSuffixes[i];
        computeElementOffset += `        {\n`;
        computeElementOffset += `            let coord = elem_rem / ${suffix}u;\n`;
        computeElementOffset += `            elem_rem = elem_rem % ${suffix}u;\n`;
        computeElementOffset += `            input_elem_offset += i32(coord) * INPUT_STRIDES[${d}];\n`;
        computeElementOffset += `        }\n`;
    }
    computeElementOffset += `        let input_offset = u32(input_elem_offset);`;

    return {
        constants: [stridesConst, shapeConst, offsetConst].join('\n'),
        computeBatchOffset,
        computeElementOffset,
    };
}

export { buildSoftmaxShader, buildLayerNormShader, buildRMSNormShader, buildBatchNormShader, buildGroupNormShader, buildLpNormalizeShader };
