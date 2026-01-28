/**
 * Window Kernel Types - Conv/Pool 操作类型定义
 * 
 * 定义 Window 操作（卷积、池化）kernel 的内部类型
 * 
 * 注意：这些类型必须与 kandle/dispatch/handlers/window.ts 中的定义保持同步
 */

import type { DType, ITensorHandle, MemoryFormat } from '@kandle/types';

// ============================================================================
// Conv/Pool Dispatch Types (Mirror from kandle)
// ============================================================================

/**
 * Conv 操作类型
 */
export type ConvVariant =
    | 'conv1d' | 'conv2d' | 'conv3d'
    | 'conv_transpose2d' | 'conv_transpose3d';

/**
 * Pool 操作类型
 */
export type PoolVariant =
    | 'max_pool1d' | 'max_pool2d' | 'max_pool3d'
    | 'avg_pool1d' | 'avg_pool2d' | 'avg_pool3d'
    | 'adaptive_avg_pool2d' | 'adaptive_max_pool2d';

/**
 * 卷积算法选择
 */
export type ConvAlgorithm = 'im2col' | 'direct' | 'winograd' | 'fft';

/**
 * Conv/Pool 操作的完整配置
 * 
 * 由 Dispatcher 构建，传递给 Backend Kernel
 */
export interface ConvDispatchResult {
    /** 操作类型 */
    variant: ConvVariant | PoolVariant;

    /** 输出张量 */
    output: ITensorHandle;

    // === 输入信息 ===
    batchSize: number;
    inChannels: number;
    outChannels: number;
    /** 输入空间维度 [H, W] 或 [L] 或 [D, H, W] */
    inputSpatial: number[];
    /** 输出空间维度 */
    outputSpatial: number[];

    // === 卷积/池化参数 ===
    kernelSize: number[];
    stride: number[];
    padding: number[];
    dilation: number[];
    groups: number;

    // === 可选输入 ===
    bias?: ITensorHandle;

    // === Pooling 特有字段 ===
    /**
     * 是否返回最大值索引 (仅 max_pool)
     * 
     * 当为 true 时，返回值为 [output, indices] 元组
     * indices 包含输入中最大值的扁平化索引
     */
    returnIndices?: boolean;

    /**
     * 索引输出张量 (当 returnIndices=true 时)
     * 
     * 形状与 output 相同，dtype 为 int64
     * 值为输入张量的扁平化索引
     */
    indicesOutput?: ITensorHandle;

    // === 算法选择 (仅 Conv) ===
    algorithm?: ConvAlgorithm;

    /** 计算类型 */
    computeDtype: DType;

    // === MemoryFormat ===
    inputFormat: MemoryFormat;
    outputFormat: MemoryFormat;
    isChannelsLast: boolean;
}

// ============================================================================
// Im2Col 配置
// ============================================================================

/**
 * Im2Col 中间张量配置
 * 
 * Im2Col 将输入转换为矩阵形式，用于 GEMM
 * 输出形状: [N * H_out * W_out, C_in * kH * kW]
 */
export interface Im2ColConfig {
    /** 输入张量 */
    input: ITensorHandle;

    /** 输出矩阵（Im2Col 结果） */
    output: ITensorHandle;

    // === 形状信息 ===
    batchSize: number;
    inChannels: number;
    inHeight: number;
    inWidth: number;
    outHeight: number;
    outWidth: number;

    // === 卷积参数 ===
    kernelH: number;
    kernelW: number;
    strideH: number;
    strideW: number;
    padH: number;
    padW: number;
    dilationH: number;
    dilationW: number;
    groups: number;

    /** 输入内存格式 */
    inputFormat: MemoryFormat;

    /** 计算类型 */
    computeDtype: DType;
}

/**
 * Im2Col Row 配置（用于 shader uniform）
 */
export interface Im2ColRowConfig {
    /** M = N * H_out * W_out */
    M: number;
    /** K = C_in_per_group * kH * kW */
    K: number;
    /** 每组的通道数 */
    channelsPerGroup: number;
}

// ============================================================================
// Conv GEMM 配置
// ============================================================================

/**
 * Conv GEMM 配置
 * 
 * 卷积作为 GEMM 执行：
 * Output = Im2Col(Input) @ Weight^T + Bias
 * 
 * 其中:
 * - Im2Col(Input): [M, K] 其中 M = N * H_out * W_out, K = C_in * kH * kW
 * - Weight^T: [K, C_out]
 * - Bias: [C_out]
 * - Output: [M, C_out]，需要 reshape 为 [N, H_out, W_out, C_out]
 */
export interface ConvGemmConfig {
    /** Im2Col 结果矩阵 [M, K] */
    im2colMatrix: ITensorHandle;

    /** 权重矩阵 [C_out, K] (需要转置使用) */
    weight: ITensorHandle;

    /** 可选偏置 [C_out] */
    bias?: ITensorHandle;

    /** 输出矩阵 [M, C_out] */
    output: ITensorHandle;

    // === 矩阵维度 ===
    M: number;  // N * H_out * W_out
    K: number;  // C_in_per_group * kH * kW
    N: number;  // C_out_per_group

    /** 分组数 */
    groups: number;

    /** 计算类型 */
    computeDtype: DType;
}

// ============================================================================
// Pool 配置
// ============================================================================

/**
 * Pool Kernel 配置
 */
export interface PoolKernelConfig {
    /** 池化类型 */
    poolType: 'max' | 'avg';

    // === 形状信息 ===
    batchSize: number;
    channels: number;
    inputH: number;
    inputW: number;
    outputH: number;
    outputW: number;

    // === 池化参数 ===
    kernelH: number;
    kernelW: number;
    strideH: number;
    strideW: number;
    padH: number;
    padW: number;
    dilationH: number;
    dilationW: number;

    /** 内存格式 */
    memoryFormat: MemoryFormat;

    /** 是否返回索引 (仅 max pool) */
    returnIndices: boolean;

    /** 是否计算填充区域 (仅 avg pool) */
    countIncludePad: boolean;

    /** 计算类型 */
    computeDtype: DType;
}

// ============================================================================
// Shader Cache
// ============================================================================

/**
 * Window Kernel 缓存键
 */
export function computeIm2ColCacheKey(
    dtype: DType,
    batchSize: number,
    inChannels: number,
    inH: number,
    inW: number,
    kernelH: number,
    kernelW: number,
    strideH: number,
    strideW: number,
    padH: number,
    padW: number,
    dilationH: number,
    dilationW: number,
    groups: number,
    isChannelsLast: boolean
): string {
    return `im2col-${dtype}-${batchSize}-${inChannels}-${inH}x${inW}-k${kernelH}x${kernelW}-s${strideH}x${strideW}-p${padH}x${padW}-d${dilationH}x${dilationW}-g${groups}-${isChannelsLast ? 'nhwc' : 'nchw'}`;
}

export function computePoolCacheKey(
    poolType: 'max' | 'avg',
    dtype: DType,
    batchSize: number,
    channels: number,
    inputH: number,
    inputW: number,
    kernelH: number,
    kernelW: number,
    strideH: number,
    strideW: number,
    padH: number,
    padW: number,
    isChannelsLast: boolean
): string {
    return `pool-${poolType}-${dtype}-${batchSize}-${channels}-${inputH}x${inputW}-k${kernelH}x${kernelW}-s${strideH}x${strideW}-p${padH}x${padW}-${isChannelsLast ? 'nhwc' : 'nchw'}`;
}

/**
 * Winograd Kernel 缓存键
 */
export function computeWinogradCacheKey(
    dtype: DType,
    batchSize: number,
    inChannels: number,
    outChannels: number,
    inH: number,
    inW: number,
    isChannelsLast: boolean
): string {
    return `winograd-${dtype}-${batchSize}-${inChannels}-${outChannels}-${inH}x${inW}-${isChannelsLast ? 'nhwc' : 'nchw'}`;
}

