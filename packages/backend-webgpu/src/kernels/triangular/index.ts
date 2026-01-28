/**
 * Triangular Kernel Registration
 * 
 * 注册 triu, tril 操作到 WebGPU 后端
 * 使用 DirectContext 模式
 */

import type { IBackendOpsRegister, DirectContext } from '@kandle/types';
import { executeTriangular } from './executor';
import { TRIANGULAR_OPS } from './ops';

/**
 * 注册 Triangular kernels
 */
export function registerTriangularKernels(registry: IBackendOpsRegister): void {
    for (const dispatchKey of Object.keys(TRIANGULAR_OPS)) {
        registry.register(dispatchKey, (ctx: DirectContext) => {
            executeTriangular(ctx, dispatchKey);
        });
    }
}
