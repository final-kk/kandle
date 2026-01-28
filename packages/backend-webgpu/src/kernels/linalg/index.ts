/**
 * Linear Algebra Kernels
 * 
 * 线性代数相关操作
 * 
 * 包含:
 * - norm: Lp 范数
 * - (future) diagonal: 对角线操作
 * 
 * 对齐: PyTorch ATen/native/LinearAlgebra.cpp
 */

export { registerNormKernels, executeNorm, NORM_OPS } from './norm';
export type { NormOpConfig, NormOrd } from './norm';
export { getNormType, normalizeOrd } from './norm';

import type { IBackendOpsRegister } from '@kandle/types';
import { registerNormKernels } from './norm';

/**
 * 注册所有 Linear Algebra kernels
 */
export function registerLinalgKernels(registry: IBackendOpsRegister): void {
    registerNormKernels(registry);
    // 未来添加: registerDiagonalKernels(registry);
}
