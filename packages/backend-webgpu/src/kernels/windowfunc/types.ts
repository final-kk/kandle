/**
 * WindowFunc Kernel Types
 * 
 * Type definitions for window function generation kernels.
 */

import type { ITensorHandle } from '@kandle/types';

/**
 * Window function template types
 */
export type WindowTemplate =
    | 'generalized_cosine'
    | 'linear'
    | 'kaiser'
    | 'gaussian'
    | 'exponential';

/**
 * Generalized Cosine Window Configuration
 * 
 * Formula: w[n] = a₀ - a₁·cos(2π·n/M) + a₂·cos(4π·n/M) - a₃·cos(6π·n/M)
 * 
 * Covers: hann, hamming, blackman, nuttall, blackman_harris
 */
export interface GeneralizedCosineConfig {
    template: 'generalized_cosine';
    coeffs: [number, number, number, number];  // [a₀, a₁, a₂, a₃]
}

/**
 * Linear Window Configuration
 * 
 * Covers: bartlett, triang
 */
export interface LinearConfig {
    template: 'linear';
    windowType: 'bartlett' | 'triang';
}

/**
 * Kaiser Window Configuration
 * 
 * Formula: w[n] = I₀(β·√(1 - ((n - M/2) / (M/2))²)) / I₀(β)
 */
export interface KaiserConfig {
    template: 'kaiser';
    beta: number;
}

/**
 * Gaussian Window Configuration
 * 
 * Formula: w[n] = exp(-0.5 * ((n - (M-1)/2) / (σ·(M-1)/2))²)
 */
export interface GaussianConfig {
    template: 'gaussian';
    std: number;
}

/**
 * Exponential Window Configuration
 * 
 * Formula: w[n] = exp(-|n - center| / τ)
 */
export interface ExponentialConfig {
    template: 'exponential';
    tau: number;
    center?: number;
}

/**
 * Union of all window configurations
 */
export type WindowConfig =
    | GeneralizedCosineConfig
    | LinearConfig
    | KaiserConfig
    | GaussianConfig
    | ExponentialConfig;

/**
 * WindowFunc Kernel Execution Arguments
 */
export interface WindowFuncKernelArgs {
    /** Output tensor handle */
    output: ITensorHandle;
    /** Window length (number of points) */
    windowLength: number;
    /** Denominator for normalization (computed from periodic/sym parameter) */
    denominator: number;
    /** Window configuration */
    config: WindowConfig;
}
