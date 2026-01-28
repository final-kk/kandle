/**
 * Gather Kernels - Types
 * 
 * 索引选择操作的类型定义
 */

import type { DType, ITensorHandle } from '@kandle/types';

/**
 * index_select kernel 参数
 */
export interface IndexSelectParams {
    /** 选择维度 (已标准化) */
    dim: number;
}

/**
 * index_select shader 参数
 * 
 * 支持非连续 (strided) 输入的工业级实现
 */
export interface IndexSelectShaderParams {
    /** 输入张量形状 */
    inputShape: readonly number[];
    /** 输入张量步幅 (用于 strided 访问) */
    inputStrides: readonly number[];
    /** 输入张量偏移量 */
    inputOffset: number;
    /** 索引张量长度 */
    indexLength: number;
    /** 索引张量步幅 (index 是 1D，步幅是标量) */
    indexStride: number;
    /** 索引张量偏移量 */
    indexOffset: number;
    /** 输出张量形状 */
    outputShape: readonly number[];
    /** 输出张量步幅 (输出总是连续的) */
    outputStrides: readonly number[];
    /** 选择维度 */
    dim: number;
    /** 数据类型 */
    dtype: DType;
    /** 输出元素数量 */
    outputSize: number;
}
