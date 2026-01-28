/**
 * Sort Kernels Registration (v5)
 *
 * Registers all sort operations with the WebGPU backend:
 * - topk: Return k largest/smallest elements
 * - sort: Sort all elements along a dimension
 * - argsort: Return indices that would sort the tensor
 */

import type { IBackendOpsRegister } from '@kandle/types';
import { topkKernel, sortKernel, argsortKernel } from './executor';

/**
 * Register all sort kernels with the backend
 */
export function registerSortKernels(registry: IBackendOpsRegister): void {
    // Register topk
    registry.register('topk', topkKernel);

    // Register sort
    registry.register('sort', sortKernel);

    // Register argsort
    registry.register('argsort', argsortKernel);
}

// Re-export types and utilities for external use
export { topkKernel, sortKernel, argsortKernel } from './executor';
export { SORT_OPS } from './ops';
export type {
    SortOpConfig,
    SortConfig,
    SortScalarArgs,
    SortOutputs,
    SortKernelImpl,
    SortAlgorithm,
    SortReturnMode,
} from './types';
