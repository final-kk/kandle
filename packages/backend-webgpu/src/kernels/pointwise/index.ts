/**
 * v5 Pointwise Kernels Registration
 */

import type { IBackendOpsRegister, ITensorIterator } from '@kandle/types';
import { executePointwise } from './executor';
import { POINTWISE_OPS } from './ops';

/**
 * Register all pointwise kernels
 */
export function registerPointwiseKernels(registry: IBackendOpsRegister): void {
    // Register all operations from POINTWISE_OPS
    for (const dispatchKey of Object.keys(POINTWISE_OPS)) {
        registry.register(dispatchKey, (iter: ITensorIterator) => {
            executePointwise(iter, dispatchKey);
        });
    }
}
