/**
 * Flip Kernel Types
 */

export interface FlipParams {
    /** Input tensor shape */
    inputShape: number[];
    /** Input tensor strides */
    inputStrides: number[];
    /** Dimensions to flip (normalized, non-negative) */
    flipDims: number[];
    /** Tensor rank */
    rank: number;
    /** Input offset in elements */
    inputOffset: number;
    /** Output offset in elements */
    outputOffset: number;
    /** Total output elements */
    numel: number;
}
