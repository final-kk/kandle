/**
 * Copy Kernel Types
 * 
 * Handles memory copy operations:
 * - cast: type conversion
 * - contiguous: strided to contiguous conversion
 */

import { DType } from '@kandle/types';

/**
 * Copy operation configuration
 */
export interface CopyOpConfig {
    /**
     * Input tensor dtype
     */
    readonly inputDtype: DType;

    /**
     * Output tensor dtype
     */
    readonly outputDtype: DType;

    /**
     * Tensor shape
     */
    readonly shape: readonly number[];

    /**
     * Input strides
     */
    readonly inputStrides: readonly number[];

    /**
     * Output strides
     */
    readonly outputStrides: readonly number[];

    /**
     * Input offset
     */
    readonly inputOffset: number;

    /**
     * Output offset
     */
    readonly outputOffset: number;

    /**
     * Total number of elements
     */
    readonly numel: number;
}

/**
 * Copy kernel variant
 */
export type CopyVariant = 'cast' | 'contiguous' | 'clone' | 'copy_';
