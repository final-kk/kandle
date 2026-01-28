/**
 * Copy Kernels Registration
 * 
 * Registers all copy operations to the backend
 */

import { IBackendOpsRegister, ITensorIterator } from '@kandle/types';
import { executeCopy } from './executor';

/**
 * Register all copy kernels
 */
export function registerCopyKernels(registry: IBackendOpsRegister): void {
    // Cast: type conversion
    registry.register('cast', (iter: ITensorIterator) => {
        executeCopy(iter, 'cast');
    });

    // Contiguous: strided to contiguous conversion
    registry.register('contiguous', (iter: ITensorIterator) => {
        executeCopy(iter, 'contiguous');
    });

    // Clone: direct copy (same as contiguous)
    registry.register('clone', (iter: ITensorIterator) => {
        executeCopy(iter, 'clone');
    });

    // copy_: in-place copy (write src to self, self can be non-contiguous view)
    registry.register('copy_', (iter: ITensorIterator) => {
        executeCopy(iter, 'copy_');
    });
}

