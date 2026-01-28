/**
 * Diff Kernel Types
 * 
 * Type definitions for N-order forward difference operation.
 */

/**
 * Uniform buffer layout for diff kernel
 */
export interface DiffUniforms {
    /** Total output elements */
    numel: number;
    /** Difference order (default 1) */
    n: number;
    /** Difference dimension */
    dim: number;
    /** Number of dimensions */
    rank: number;
    /** Input offset in elements */
    inputOffset: number;
    /** Output offset in elements */
    outputOffset: number;
    /** Input shape (padded to 8) */
    inputShape: number[];
    /** Input strides (padded to 8) */
    inputStrides: number[];
    /** Output shape (padded to 8) */
    outputShape: number[];
    /** Output strides (padded to 8) */
    outputStrides: number[];
}

/**
 * Parameters passed from handler
 */
export interface DiffParams {
    n: number;
    dim: number;
}
