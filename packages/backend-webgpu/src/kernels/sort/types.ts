/**
 * Sort Kernels - Type Definitions (v5)
 *
 * Defines types for sort operations: topk, sort, argsort
 * Follows DirectContext pattern (similar to MatrixOp, not TensorIterator)
 */

import type { DType, ITensorHandle } from '@kandle/types';

/**
 * Sort algorithm types
 * 
 * - bitonic: GPU-friendly O(n log²n) parallel sort
 * - radix_select: O(n + k log k) for topk (future optimization)
 * - radix_sort: O(n·d) where d = bits (future optimization)
 */
export type SortAlgorithm = 'bitonic' | 'radix_select' | 'radix_sort';

/**
 * Return mode for sort operations
 */
export type SortReturnMode = 'tuple' | 'values' | 'indices';

/**
 * Scalar parameter names for sort operations
 */
export type SortScalarParam =
    | 'k'           // topk: number of elements to return
    | 'dim'         // dimension to sort along
    | 'largest'     // topk: whether to return largest (true) or smallest (false)
    | 'sorted'      // topk: whether outputs should be sorted
    | 'descending'  // sort/argsort: sort direction
    | 'stable';     // sort/argsort: whether sort is stable

/**
 * Sort operation configuration
 */
export interface SortOpConfig {
    /** Algorithm to use for sorting */
    readonly algorithm: SortAlgorithm;

    /** Return mode: tuple (values, indices), values only, or indices only */
    readonly returns: SortReturnMode;

    /** Whether this operation produces indices output */
    readonly needsIndices: boolean;

    /** Whether this operation produces values output */
    readonly needsValues: boolean;

    /** List of scalar parameter names expected by this operation */
    readonly scalarParams: readonly SortScalarParam[];

    /** Default values for scalar parameters */
    readonly scalarDefaults: Partial<Record<SortScalarParam, number | boolean>>;
}

/**
 * Processed scalar arguments for sort execution
 */
export interface SortScalarArgs {
    /** Number of elements to return (topk only) */
    k?: number;

    /** Dimension to sort along (-1 means last dimension) */
    dim: number;

    /** Whether to return largest elements (topk: true, sort: descending) */
    largest: boolean;

    /** Whether output should be sorted (topk) */
    sorted: boolean;

    /** Whether sort should be stable */
    stable: boolean;

    /** Descending order flag (sort/argsort) */
    descending: boolean;
}

/**
 * Sort execution configuration
 * Contains all computed parameters needed for GPU execution
 */
export interface SortConfig {
    /** Input tensor handle */
    readonly input: ITensorHandle;

    /** Operation configuration from registry */
    readonly opConfig: SortOpConfig;

    /** Dispatch key (topk, sort, argsort) */
    readonly dispatchKey: string;

    /** Processed scalar arguments */
    readonly scalars: SortScalarArgs;

    /** Normalized dimension (positive, within bounds) */
    readonly dim: number;

    /** Size of the dimension being sorted */
    readonly dimSize: number;

    /** Number of independent "slices" to sort in parallel */
    readonly numSlices: number;

    /** Output tensor shape */
    readonly outputShape: readonly number[];

    /** Output element count (k for topk, dimSize for full sort) */
    readonly outputDimSize: number;

    /** Input dtype */
    readonly dtype: DType;
}

/**
 * Sort kernel outputs
 */
export interface SortOutputs {
    /** Values tensor (for topk/sort) */
    values?: ITensorHandle;

    /** Indices tensor (for all operations) */
    indices?: ITensorHandle;
}

/**
 * Kernel signature for sort operations
 * 
 * Unlike TensorIterator-based kernels, sort kernels receive:
 * - input: Single input tensor
 * - scalars: Map of scalar parameters
 * - outs: Optional pre-allocated output tensors
 * 
 * Returns single tensor (argsort) or tuple (topk, sort)
 */
export type SortKernelImpl = (
    input: ITensorHandle,
    scalars: Record<string, unknown>,
    outs?: ITensorHandle[]
) => ITensorHandle | [ITensorHandle, ITensorHandle];

/**
 * TopK-specific kernel signature
 */
export type TopKKernelImpl = (
    input: ITensorHandle,
    scalars: Record<string, unknown>,
    outs?: ITensorHandle[]
) => [ITensorHandle, ITensorHandle];

/**
 * ArgSort-specific kernel signature
 */
export type ArgSortKernelImpl = (
    input: ITensorHandle,
    scalars: Record<string, unknown>,
    outs?: ITensorHandle[]
) => ITensorHandle;
