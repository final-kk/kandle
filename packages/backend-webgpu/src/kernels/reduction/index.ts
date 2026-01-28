/**
 * Reduction Kernels Registration (v5)
 * 
 * Registers all reduction operations with the backend
 */

import { IBackendOpsRegister, ITensorIterator } from '@kandle/types';
import { executeReduction } from './executor';
import { executeArgReduction } from './argExecutor';
import { REDUCTION_OPS } from './ops';

export function registerReductionKernels(registry: IBackendOpsRegister): void {
    // Register all reduction operations from REDUCTION_OPS
    for (const dispatchKey of Object.keys(REDUCTION_OPS)) {
        registry.register(dispatchKey, (iter: ITensorIterator) => {
            executeReduction(iter, dispatchKey);
        });
    }

    // Register argmax/argmin (special handling - only output indices)
    registry.register('argmax', (iter: ITensorIterator) => {
        executeArgReduction(iter, 'argmax');
    });

    registry.register('argmin', (iter: ITensorIterator) => {
        executeArgReduction(iter, 'argmin');
    });
}

export { REDUCTION_OPS } from './ops';
export { executeReduction } from './executor';
export { executeArgReduction } from './argExecutor';
export type { ReductionOpConfig } from './types';
