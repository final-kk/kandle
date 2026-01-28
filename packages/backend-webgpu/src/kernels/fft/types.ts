/**
 * FFT Kernel Types
 * 
 * Type definitions for FFT operations
 */

import type { ITensorHandle } from '@kandle/types';

/**
 * FFT normalization mode - matches PyTorch's norm parameter
 * - 'backward': No normalization on forward, 1/N on inverse (default)
 * - 'forward': 1/N on forward, no normalization on inverse
 * - 'ortho': 1/sqrt(N) on both forward and inverse
 */
export type FFTNorm = 'forward' | 'backward' | 'ortho';

/**
 * FFT direction
 */
export type FFTDirection = 'forward' | 'inverse';

/**
 * FFT Kernel Arguments
 */
export interface FFTKernelArgs {
    /** Input tensor (complex or real) */
    input: ITensorHandle;

    /** Output tensor (complex) */
    output: ITensorHandle;

    /** FFT dimension (resolved, non-negative) */
    dim: number;

    /** FFT size (must be power of 2) */
    n: number;

    /** Normalization mode */
    norm: FFTNorm;

    /** Transform direction */
    direction: FFTDirection;

    /** Whether input is real (will be zero-padded to complex) */
    isRealInput: boolean;
}

/**
 * FFT Stage Parameters for butterfly shader
 */
export interface FFTStageParams {
    /** Current stage (0 to log2(N)-1) */
    stage: number;
    /** FFT size */
    fftSize: number;
    /** Transform direction */
    direction: FFTDirection;
}

/**
 * RFFT Kernel Arguments (Real to Complex, onesided)
 */
export interface RFFTKernelArgs {
    /** Input tensor (real) */
    input: ITensorHandle;

    /** Output tensor (complex, onesided: n//2+1) */
    output: ITensorHandle;

    /** FFT dimension (resolved, non-negative) */
    dim: number;

    /** Signal length (input size along dim) */
    n: number;

    /** Normalization mode */
    norm: FFTNorm;

    /** Output length (= n//2 + 1) */
    onesidedLen: number;
}

/**
 * IRFFT Kernel Arguments (Complex onesided to Real)
 */
export interface IRFFTKernelArgs {
    /** Input tensor (complex, onesided: n//2+1) */
    input: ITensorHandle;

    /** Output tensor (real) */
    output: ITensorHandle;

    /** FFT dimension (resolved, non-negative) */
    dim: number;

    /** Output signal length (full FFT size) */
    n: number;

    /** Normalization mode */
    norm: FFTNorm;

    /** Input onesided length (= n//2 + 1) */
    onesidedLen: number;
}

