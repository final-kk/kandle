/**
 * Scatter Kernels - Types
 *
 * 散射操作的类型定义
 */

import type { DType, ITensorHandle } from '@kandle/types';

/**
 * Scatter 操作类型
 */
export type ScatterOp = 'scatter' | 'scatter_add' | 'scatter_reduce';

/**
 * Scatter Reduce 归约模式
 */
export type ScatterReduceMode = 'sum' | 'prod' | 'mean' | 'amax' | 'amin';

/**
 * scatter kernel 参数
 */
export interface ScatterParams {
    /** 散射维度 (已标准化) */
    dim: number;
}

/**
 * scatter_add kernel 参数
 */
export interface ScatterAddParams extends ScatterParams { }

/**
 * scatter_reduce kernel 参数
 */
export interface ScatterReduceParams extends ScatterParams {
    /** 归约模式 */
    reduce: ScatterReduceMode;
    /** 是否包含 self 原值参与归约 */
    includeSelf: boolean;
}

/**
 * Scatter Shader 配置
 */
export interface ScatterOpConfig {
    /** 操作名称 */
    name: ScatterOp;
    /** 是否需要原子操作 */
    needsAtomic: boolean;
    /** 归约模式 (仅 scatter_reduce 有效) */
    reduceMode?: ScatterReduceMode;
    /** 是否包含 self 原值 */
    includeSelf?: boolean;
}

/**
 * Scatter Shader 参数
 * 
 * 工业级实现：原生支持 strided (非连续) 输入
 */
export interface ScatterShaderParams {
    /** 操作配置 */
    config: ScatterOpConfig;
    /** self 张量形状 */
    selfShape: readonly number[];
    /** self 张量步幅 (用于 strided 访问) */
    selfStrides: readonly number[];
    /** self 张量偏移量 */
    selfOffset: number;
    /** index 张量形状 */
    indexShape: readonly number[];
    /** index 张量步幅 (用于 strided 访问) */
    indexStrides: readonly number[];
    /** index 张量偏移量 */
    indexOffset: number;
    /** src 张量形状 */
    srcShape: readonly number[];
    /** src 张量步幅 (用于 strided 访问) */
    srcStrides: readonly number[];
    /** src 张量偏移量 */
    srcOffset: number;
    /** 输出张量形状 */
    outputShape: readonly number[];
    /** 输出张量步幅 (输出总是连续的) */
    outputStrides: readonly number[];
    /** 散射维度 */
    dim: number;
    /** 数据类型 */
    dtype: DType;
    /** index 元素数量 (遍历量) */
    indexSize: number;
    /** 输出元素数量 */
    outputSize: number;
}

/**
 * Scatter Kernel 签名
 */
export type ScatterKernelImpl = (
    self: ITensorHandle,
    index: ITensorHandle,
    src: ITensorHandle,
    params: Record<string, unknown>,
    output: ITensorHandle
) => void;
