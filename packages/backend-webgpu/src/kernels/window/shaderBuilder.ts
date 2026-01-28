/**
 * Window Kernel Shader Builder (工业级实现)
 * 
 * Pool 操作的 WGSL Shader 生成器
 * 
 * 工业级特性：
 * - 输入使用 shape/strides/offset 计算物理地址，支持非连续访问
 * - 输出总是连续的 (新分配的 buffer)
 * 
 * 职责：
 * 1. 生成 Pool2d/Pool1d/Pool3d 的 WGSL shader 代码
 * 2. 处理不同的内存布局 (NCHW/NHWC)
 * 3. 支持 max/avg pooling 的不同初始化和归约操作
 * 
 * @module kernels/window/shaderBuilder
 */

import type { DType } from '@kandle/types';
import { WGSL_CONSTANTS } from '../../base/dtype';

// ============================================================================
// Constants
// ============================================================================

export const POOL_WORKGROUP_SIZE_X = 8;
export const POOL_WORKGROUP_SIZE_Y = 8;

// ============================================================================
// Pool2d Shader
// ============================================================================

export interface Pool2dShaderConfig {
    batchSize: number;
    channels: number;
    H: number;
    W: number;
    outH: number;
    outW: number;
    kH: number;
    kW: number;
    strideH: number;
    strideW: number;
    padH: number;
    padW: number;
    dilationH: number;
    dilationW: number;
    isMaxPool: boolean;
    isChannelsLast: boolean;
    dtype: DType;
    /** 是否输出最大值索引 (仅 max_pool) */
    returnIndices?: boolean;
    // === 工业级 Strided 支持 ===
    /** 输入 tensor 的 strides (4D: [N, C, H, W] 或 [N, H, W, C]) */
    inputStrides: readonly number[];
    /** 输入 tensor 的 offset */
    inputOffset: number;
}

/**
 * 生成 Pool2d WGSL shader 代码 (工业级: Strided 输入支持)
 * 
 * 支持功能：
 * - Max/Avg pooling
 * - NCHW/NHWC 内存格式
 * - returnIndices (仅 max_pool)
 * - 非连续输入：通过 strides/offset 计算物理地址
 */
export function buildPool2dShader(config: Pool2dShaderConfig): string {
    const {
        isMaxPool, isChannelsLast, dtype, returnIndices,
        inputStrides, inputOffset
    } = config;

    const dataType = dtype === 'float16' ? 'f16' : 'f32';
    const enableF16 = dtype === 'float16' ? 'enable f16;' : '';

    // 对于 max pool，初始化为 -FLT_MAX；对于 avg pool，初始化为 0
    const initValue = isMaxPool
        ? `${dataType}(${WGSL_CONSTANTS.NEG_FLT_MAX})`
        : `${dataType}(0.0)`;

    // 工业级: 生成 strided 输入索引函数
    const inputIndexFn = generateStridedInputIndexFn(isChannelsLast, inputStrides, inputOffset);
    const outputIndexFn = generateOutputIndexFn(isChannelsLast);

    // indices 输出缓冲区绑定 (仅当 returnIndices=true)
    // 注意：WebGPU 不支持 i64 storage，使用 i32（对于大多数 tensor 足够）
    const indicesBinding = returnIndices
        ? '@group(0) @binding(3) var<storage, read_write> indices: array<i32>;'
        : '';

    // Max pool with indices tracking
    if (isMaxPool && returnIndices) {
        return `
${enableF16}

struct Uniforms {
    batchSize: i32,
    channels: i32,
    H: i32,
    W: i32,
    outH: i32,
    outW: i32,
    kH: i32,
    kW: i32,
    strideH: i32,
    strideW: i32,
    padH: i32,
    padW: i32,
    dilationH: i32,
    dilationW: i32,
    _padding1: i32,
    _padding2: i32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<${dataType}>;
@group(0) @binding(2) var<storage, read_write> output: array<${dataType}>;
${indicesBinding}

${inputIndexFn}
${outputIndexFn}

@compute @workgroup_size(${POOL_WORKGROUP_SIZE_X}, ${POOL_WORKGROUP_SIZE_Y}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w_out = i32(gid.x);
    let h_out = i32(gid.y);
    let idx = i32(gid.z);
    
    if (w_out >= uniforms.outW || h_out >= uniforms.outH) {
        return;
    }
    
    let n = idx / uniforms.channels;
    let c = idx % uniforms.channels;
    
    if (n >= uniforms.batchSize) {
        return;
    }
    
    var result: ${dataType} = ${initValue};
    var maxIdx: i32 = -1;  // 追踪最大值的扁平化索引
    
    for (var kh: i32 = 0; kh < uniforms.kH; kh++) {
        for (var kw: i32 = 0; kw < uniforms.kW; kw++) {
            let h_in = h_out * uniforms.strideH - uniforms.padH + kh * uniforms.dilationH;
            let w_in = w_out * uniforms.strideW - uniforms.padW + kw * uniforms.dilationW;
            
            if (h_in >= 0 && h_in < uniforms.H && w_in >= 0 && w_in < uniforms.W) {
                let inputIdx = getInputIndex(n, c, h_in, w_in);
                let value = input[inputIdx];
                if (value > result) {
                    result = value;
                    maxIdx = inputIdx;
                }
            }
        }
    }
    
    let outIdx = getOutputIndex(n, c, h_out, w_out);
    output[outIdx] = result;
    indices[outIdx] = maxIdx;
}
`;
    }

    // Standard max/avg pool (no indices)
    const reduceOp = isMaxPool ? 'max(result, value)' : 'result + value';
    const avgPoolFinalize = isMaxPool
        ? ''
        : `if (count > 0) { result = result / ${dataType}(count); }`;

    return `
${enableF16}

struct Uniforms {
    batchSize: i32,
    channels: i32,
    H: i32,
    W: i32,
    outH: i32,
    outW: i32,
    kH: i32,
    kW: i32,
    strideH: i32,
    strideW: i32,
    padH: i32,
    padW: i32,
    dilationH: i32,
    dilationW: i32,
    _padding1: i32,
    _padding2: i32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<${dataType}>;
@group(0) @binding(2) var<storage, read_write> output: array<${dataType}>;

${inputIndexFn}
${outputIndexFn}

@compute @workgroup_size(${POOL_WORKGROUP_SIZE_X}, ${POOL_WORKGROUP_SIZE_Y}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w_out = i32(gid.x);
    let h_out = i32(gid.y);
    let idx = i32(gid.z);
    
    if (w_out >= uniforms.outW || h_out >= uniforms.outH) {
        return;
    }
    
    let n = idx / uniforms.channels;
    let c = idx % uniforms.channels;
    
    if (n >= uniforms.batchSize) {
        return;
    }
    
    var result: ${dataType} = ${initValue};
    var count: i32 = 0;
    
    for (var kh: i32 = 0; kh < uniforms.kH; kh++) {
        for (var kw: i32 = 0; kw < uniforms.kW; kw++) {
            let h_in = h_out * uniforms.strideH - uniforms.padH + kh * uniforms.dilationH;
            let w_in = w_out * uniforms.strideW - uniforms.padW + kw * uniforms.dilationW;
            
            if (h_in >= 0 && h_in < uniforms.H && w_in >= 0 && w_in < uniforms.W) {
                let value = input[getInputIndex(n, c, h_in, w_in)];
                result = ${reduceOp};
                count = count + 1;
            }
        }
    }
    
    ${avgPoolFinalize}
    
    output[getOutputIndex(n, c, h_out, w_out)] = result;
}
`;
}


// ============================================================================
// Index Function Generators (工业级: Strided 支持)
// ============================================================================

/**
 * 生成 strided 输入索引计算函数 (工业级)
 * 
 * 使用 strides 和 offset 计算物理地址，支持任意非连续输入
 * 
 * 重要：strides 数组始终按逻辑形状维度顺序存储 [strideN, strideC, strideH, strideW]，
 * 与 memoryFormat 无关。memoryFormat 影响的是 stride 的值，而不是 stride 在数组中的位置。
 * 
 * @param isChannelsLast - 这个参数在此函数中实际上不需要使用，保留仅为了 API 兼容性
 * @param strides - 输入 tensor 的 strides，始终为 [strideN, strideC, strideH, strideW]
 * @param offset - 输入 tensor 的 offset
 */
function generateStridedInputIndexFn(
    isChannelsLast: boolean,
    strides: readonly number[],
    offset: number
): string {
    // 4D tensor strides 始终按逻辑形状顺序存储: [strideN, strideC, strideH, strideW]
    // memoryFormat 影响的是 stride 的值:
    // - NCHW: strides 可能是 [C*H*W, H*W, W, 1] (C 维度跨度大)
    // - NHWC: strides 可能是 [H*W*C, 1, W*C, C] (C 维度跨度小，最密集)
    // 但无论哪种格式，strides[0] 总是 N 维度的 stride，strides[1] 总是 C 维度的 stride，以此类推

    if (strides.length !== 4) {
        throw new Error(`Pool2d requires 4D input, got strides length ${strides.length}`);
    }

    const strideN = strides[0];
    const strideC = strides[1];
    const strideH = strides[2];
    const strideW = strides[3];

    return `
// Strided 输入索引计算 (工业级: 支持非连续访问)
const INPUT_OFFSET: i32 = ${offset};
const STRIDE_N: i32 = ${strideN};
const STRIDE_C: i32 = ${strideC};
const STRIDE_H: i32 = ${strideH};
const STRIDE_W: i32 = ${strideW};

fn getInputIndex(n: i32, c: i32, h: i32, w: i32) -> i32 {
    return INPUT_OFFSET + n * STRIDE_N + c * STRIDE_C + h * STRIDE_H + w * STRIDE_W;
}`;
}

/**
 * 生成输出索引计算函数 (连续输出)
 * 
 * 输出总是连续的，使用标准公式
 */
function generateOutputIndexFn(isChannelsLast: boolean): string {
    if (isChannelsLast) {
        return `fn getOutputIndex(n: i32, c: i32, h: i32, w: i32) -> i32 {
    return n * (uniforms.outH * uniforms.outW * uniforms.channels) +
           h * (uniforms.outW * uniforms.channels) +
           w * uniforms.channels + c;
}`;
    } else {
        return `fn getOutputIndex(n: i32, c: i32, h: i32, w: i32) -> i32 {
    return n * (uniforms.channels * uniforms.outH * uniforms.outW) +
           c * (uniforms.outH * uniforms.outW) +
           h * uniforms.outW + w;
}`;
    }
}

// ============================================================================
// Pipeline Cache Key (工业级: 包含 strided 信息)
// ============================================================================

/**
 * 生成 Pool2d pipeline 缓存键
 * 
 * 工业级: 包含 strides 信息用于区分不同的非连续输入模式
 */
export function computePool2dCacheKey(config: Pool2dShaderConfig): string {
    const {
        batchSize, channels, H, W, kH, kW,
        isMaxPool, isChannelsLast, dtype, returnIndices,
        inputStrides, inputOffset
    } = config;

    const poolType = isMaxPool ? 'max' : 'avg';
    const idxSuffix = returnIndices ? '-idx' : '';
    // 工业级: 包含 strides 和 offset
    const stridesSuffix = `-s${inputStrides.join('_')}-o${inputOffset}`;
    return `pool2d-${poolType}-${dtype}-${batchSize}-${channels}-${H}x${W}-k${kH}x${kW}-${isChannelsLast}${idxSuffix}${stridesSuffix}`;
}
