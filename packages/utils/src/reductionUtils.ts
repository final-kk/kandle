/**
 * Reduction Utilities
 * 
 * Helper functions for reduction operations (sum, mean, max, min, prod, etc.)
 * These are generic utilities that can be reused across different tensor libraries.
 */

/**
 * Normalize axis parameter to array of positive indices
 * @param axis - Can be number, number[], or undefined (全局归约)
 * @param rank - Tensor rank
 * @returns Normalized array of axes, sorted and deduplicated
 */
export function normalizeAxis(axis: number | number[] | undefined, rank: number): number[] {
    // undefined means reduce all dimensions
    if (axis === undefined) {
        return Array.from({ length: rank }, (_, i) => i);
    }

    // Convert to array
    const axes = Array.isArray(axis) ? axis : [axis];

    // Normalize negative indices and validate
    const normalized = axes.map(a => {
        const normalized = a < 0 ? rank + a : a;
        if (normalized < 0 || normalized >= rank) {
            throw new Error(`Axis ${a} is out of bounds for tensor of rank ${rank}`);
        }
        return normalized;
    });

    // Remove duplicates and sort
    return [...new Set(normalized)].sort((a, b) => a - b);
}

/**
 * Compute output shape after reduction
 * @param inputShape - Input tensor shape
 * @param axes - Axes to reduce (already normalized)
 * @param keepDims - Whether to keep reduced dimensions as size 1
 * @returns Output shape
 */
export function computeReductionShape(
    inputShape: readonly number[],
    axes: readonly number[],
    keepDims: boolean
): number[] {
    if (keepDims) {
        // Keep all dimensions, set reduced axes to 1
        return inputShape.map((dim, i) => axes.includes(i) ? 1 : dim);
    } else {
        // Remove reduced dimensions
        return inputShape.filter((_, i) => !axes.includes(i));
    }
}

/**
 * Compute shape of reduction dimensions (for inner loop)
 * @param inputShape - Input tensor shape
 * @param axes - Axes to reduce
 * @returns Shape of reduction dimensions
 */
export function computeReductionDimShape(
    inputShape: readonly number[],
    axes: readonly number[]
): number[] {
    return axes.map(axis => inputShape[axis]);
}

/**
 * Split strides into parallel and reduction components
 * @param inputShape - Input tensor shape
 * @param inputStrides - Input tensor strides
 * @param axes - Axes to reduce
 * @returns { parallelShape, parallelStrides, reductionShape, reductionStrides }
 */
export function splitStridesForReduction(
    inputShape: readonly number[],
    inputStrides: readonly number[],
    axes: readonly number[]
) {
    const parallelShape: number[] = [];
    const parallelStrides: number[] = [];
    const reductionShape: number[] = [];
    const reductionStrides: number[] = [];

    inputShape.forEach((dim, i) => {
        if (axes.includes(i)) {
            reductionShape.push(dim);
            reductionStrides.push(inputStrides[i]);
        } else {
            parallelShape.push(dim);
            parallelStrides.push(inputStrides[i]);
        }
    });

    return {
        parallelShape,
        parallelStrides,
        reductionShape,
        reductionStrides
    };
}

/**
 * Compute keepDims output strides from parallel strides
 * When keepDims=true, reduced axes have size 1 and stride 0
 * @param inputShape - Original input shape
 * @param axes - Axes being reduced
 * @param parallelStrides - Strides for non-reduced dimensions
 * @returns Output strides with stride=0 for reduced axes
 */
export function computeKeepDimsStrides(
    inputShape: readonly number[],
    axes: readonly number[],
    parallelStrides: readonly number[]
): number[] {
    const outputStrides: number[] = [];
    let parallelIdx = 0;

    for (let i = 0; i < inputShape.length; i++) {
        if (axes.includes(i)) {
            // Reduced dimension: stride = 0 (broadcast from single element)
            outputStrides.push(0);
        } else {
            // Non-reduced dimension: use parallel stride
            outputStrides.push(parallelStrides[parallelIdx]);
            parallelIdx++;
        }
    }

    return outputStrides;
}
