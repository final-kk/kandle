/**
 * Sort Operations Registry (v5)
 *
 * Centralized configuration for all sort operations: topk, sort, argsort
 * 
 * Design Decisions:
 * 1. Uses DirectContext pattern (like MatrixOp), not TensorIterator
 * 2. Algorithm selection: bitonic for Phase 1 (GPU-friendly, O(n log²n))
 * 3. Registry-driven dispatch eliminates code duplication
 * 
 * PyTorch Reference:
 * - topk uses radix select for large arrays, but bitonic is simpler for Phase 1
 * - sort dynamically selects bitonicSort or radixSort based on size
 */

import type { SortOpConfig } from './types';

/**
 * SORT_OPS - Configuration registry for all sort operations
 * 
 * Each entry defines:
 * - algorithm: sorting algorithm to use
 * - returns: what the operation returns (tuple, values, indices)
 * - needsIndices/needsValues: which outputs to allocate
 * - scalarParams: expected scalar parameters
 * - scalarDefaults: default values for optional parameters
 */
export const SORT_OPS: Record<string, SortOpConfig> = {
    /**
     * topk: Return the k largest (or smallest) elements along a dimension
     * 
     * PyTorch: torch.topk(input, k, dim=-1, largest=True, sorted=True)
     * Returns: (values, indices) - both with shape where dim is replaced with k
     */
    topk: {
        algorithm: 'bitonic',  // Phase 1: bitonic, future: radix_select for O(n + k log k)
        returns: 'tuple',
        needsIndices: true,
        needsValues: true,
        scalarParams: ['k', 'dim', 'largest', 'sorted'] as const,
        scalarDefaults: {
            dim: -1,
            largest: true,
            sorted: true,
        },
    },

    /**
     * sort: Sort elements along a dimension
     * 
     * PyTorch: torch.sort(input, dim=-1, descending=False, stable=False)
     * Returns: (values, indices) - both with same shape as input
     */
    sort: {
        algorithm: 'bitonic',  // Phase 1: bitonic, future: radix_sort for O(n·d)
        returns: 'tuple',
        needsIndices: true,
        needsValues: true,
        scalarParams: ['dim', 'descending', 'stable'] as const,
        scalarDefaults: {
            dim: -1,
            descending: false,
            stable: false,
        },
    },

    /**
     * argsort: Return indices that would sort the tensor along a dimension
     * 
     * PyTorch: torch.argsort(input, dim=-1, descending=False, stable=False)
     * Returns: indices tensor with same shape as input
     * 
     * Optimization: needsValues=false allows skipping values output allocation
     */
    argsort: {
        algorithm: 'bitonic',
        returns: 'indices',
        needsIndices: true,
        needsValues: false,  // Optimization: skip values output
        scalarParams: ['dim', 'descending', 'stable'] as const,
        scalarDefaults: {
            dim: -1,
            descending: false,
            stable: false,
        },
    },
};

/**
 * Get output dimension size for a sort operation
 * 
 * - topk: k
 * - sort/argsort: dimSize (unchanged)
 */
export function getOutputDimSize(
    dispatchKey: string,
    dimSize: number,
    k?: number
): number {
    if (dispatchKey === 'topk') {
        if (k === undefined) {
            throw new Error('topk requires k parameter');
        }
        return k;
    }
    return dimSize;
}

/**
 * Validate sort operation parameters
 * 
 * @throws Error if parameters are invalid
 */
export function validateSortParams(
    dispatchKey: string,
    dim: number,
    dimSize: number,
    k?: number
): void {
    // k validation for topk
    if (dispatchKey === 'topk') {
        if (k === undefined) {
            throw new Error('topk requires k parameter');
        }
        if (k <= 0) {
            throw new Error(`topk: k must be positive, got ${k}`);
        }
        if (k > dimSize) {
            throw new Error(
                `topk: k (${k}) cannot be larger than dimension size (${dimSize})`
            );
        }
    }

    // Dimension validation is done earlier during normalization
    // This is a secondary check
    if (dim < 0) {
        throw new Error(`Invalid normalized dimension: ${dim}`);
    }
}
