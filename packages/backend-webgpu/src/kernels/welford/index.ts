/**
 * Welford Kernels Registration
 * 
 * 注册 variance 和 std 操作到 backend
 */

import type { IBackendOpsRegister, ITensorIterator } from '@kandle/types';
import { executeWelford } from './executor';
import { WELFORD_OPS } from './ops';

/**
 * 注册所有 Welford kernels
 */
export function registerWelfordKernels(registry: IBackendOpsRegister): void {
    // 注册所有 Welford 操作
    for (const dispatchKey of Object.keys(WELFORD_OPS)) {
        registry.register(dispatchKey, (iter: ITensorIterator) => {
            executeWelford(iter, dispatchKey);
        });
    }
}

// 导出类型和工具函数
export { WELFORD_OPS } from './ops';
export { executeWelford } from './executor';
export type { WelfordOpConfig, WelfordDimParams, WelfordGlobalParams } from './types';
