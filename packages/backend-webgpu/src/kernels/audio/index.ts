/**
 * Audio Kernels
 *
 * 音频处理相关的 WebGPU kernel
 *
 * 当前包含:
 * - IIR 滤波器 (biquad)
 *
 * 未来可扩展:
 * - Resampling kernel
 * - Mel filterbank kernel
 */

import type { IBackendOpsRegister } from '@kandle/types';
import { registerIIRkernel } from './iir';

// Re-export types
export type { BiquadCoeffs, IIRScanParams, IIRBiquadKernelArgs } from './iir';

/**
 * 注册所有 Audio kernels
 */
export function registerAudioKernels(registry: IBackendOpsRegister): void {
    registerIIRkernel(registry);
}
