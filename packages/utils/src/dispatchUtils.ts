/**
 * Dispatch Utilities
 * 
 * Helper functions for tensor dispatch operations
 */

import { ITensorHandle } from "@kandle/types";
import { isShapeEquals } from "./shape";

/**
 * Detect if two tensors sharing storage have write overlap
 * 
 * Conservative strategy:
 * - If they are exactly the same view (offset/shape/strides all equal), it's safe
 * - Otherwise, assume potential overlap and report error
 * 
 * This may false-positive some actually safe cases (e.g., interleaved access),
 * but guarantees correctness. More precise overlap detection can be implemented later.
 */
export function detectOverlap(
    out: ITensorHandle,
    input: ITensorHandle
): boolean {
    // If offset/shape/strides are all equal, this is a safe in-place operation
    // (e.g., a.add(b, out=a) where a itself is the result target)
    if (out.offset === input.offset &&
        isShapeEquals(out.shape, input.shape) &&
        stridesEqual(out.strides, input.strides)) {
        return false;
    }

    // Otherwise, conservatively assume overlap
    return true;
}

/**
 * Compare two strides arrays for equality
 */
export function stridesEqual(s1: readonly number[], s2: readonly number[]): boolean {
    if (s1.length !== s2.length) return false;
    for (let i = 0; i < s1.length; i++) {
        if (s1[i] !== s2[i]) return false;
    }
    return true;
}
