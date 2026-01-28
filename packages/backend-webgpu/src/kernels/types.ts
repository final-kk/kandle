/**
 * WebGPU Kernels - Core Types
 *
 * v5 架构: 统一的 Kernel 类型定义
 */

import type { ITensorIterator } from '@kandle/types';

/**
 * Kernel 执行函数签名
 * 所有使用 TensorIterator 的 kernel 都使用这个签名
 */
export type IteratorKernelFn = (iter: ITensorIterator) => void;

/**
 * 直接 Kernel 执行函数签名
 * 不使用 TensorIterator 的 kernel
 */
export type DirectKernelFn = (params: Record<string, unknown>) => void;
