/**
 * Norm Kernel Registration
 * 
 * 注册 norm 操作到 WebGPU 后端
 */

import type { IBackendOpsRegister, ITensorIterator } from '@kandle/types';
import { executeNorm } from './executor';
import { NORM_OPS } from './ops';

/**
 * 注册 Norm kernels
 */
export function registerNormKernels(registry: IBackendOpsRegister): void {
    for (const dispatchKey of Object.keys(NORM_OPS)) {
        registry.register(dispatchKey, (iter: ITensorIterator) => {
            executeNorm(iter, dispatchKey);
        });
    }
}

// 导出模块
export { executeNorm } from './executor';
export { NORM_OPS } from './ops';
export type { NormOpConfig, NormOrd } from './types';
export { getNormType, normalizeOrd } from './types';
