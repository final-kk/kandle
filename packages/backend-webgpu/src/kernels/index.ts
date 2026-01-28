/**
 * v6 WebGPU Kernels Registration
 * 
 * Central registration point for all WebGPU backend kernels
 * 
 * 所有 kernels 通过 operators.register 注册，使用统一的 KernelImpl 接口
 * Handler 层通过 operators.find 获取 kernel 并调用
 */

import type { IBackendOpsRegister } from '@kandle/types';
import { registerPointwiseKernels } from './pointwise';
import { registerReductionKernels } from './reduction';
import { registerScanKernels } from './scan';
import { registerCopyKernels } from './copy';
import { registerMatrixKernels } from './matrix';
import { registerSortKernels } from './sort';
import { registerWindowKernels } from './window';
import { registerNormalizeKernels } from './normalize';
import { registerGatherKernels } from './gather';
import { registerScatterKernels } from './scatter';
import { registerRandomKernels } from './random';
import { registerWelfordKernels } from './welford';

import { registerLinalgKernels } from './linalg';
import { registerTriangularKernels } from './triangular';
import { registerFactoryKernels } from './factory';
import { registerAttentionKernels } from './attention';
import { registerWindowFuncKernels } from './windowfunc';
import { registerFFTKernels } from './fft';
import { registerShapeKernels } from './shape';
import { registerAudioKernels } from './audio';

/**
 * Register all WebGPU kernels
 */
export function registerWebGPUKernels(registry: IBackendOpsRegister): void {
    // Pointwise (Map operations)
    registerPointwiseKernels(registry);

    // Reduction operations
    registerReductionKernels(registry);
    registerWelfordKernels(registry);

    // Scan operations
    registerScanKernels(registry);

    // Memory operations
    registerCopyKernels(registry);

    // Matrix operations
    registerMatrixKernels(registry);
    registerLinalgKernels(registry);

    // Shape operations (Direct kernels)
    registerTriangularKernels(registry);

    // Factory operations (Direct kernels) - includes eye, arange, repeat_interleave
    registerFactoryKernels(registry);

    // Sort operations
    registerSortKernels(registry);

    // Window operations (Conv/Pool)
    registerWindowKernels(registry);

    // Normalize operations
    registerNormalizeKernels(registry);

    // Index operations
    registerGatherKernels(registry);
    registerScatterKernels(registry);

    // Random operations - includes rand, randn, randint, multinomial
    registerRandomKernels(registry);

    // Attention operations - includes flash_attention
    registerAttentionKernels(registry);

    // WindowFunc operations - includes hann, kaiser, etc. (signal processing)
    registerWindowFuncKernels(registry);

    // FFT operations - includes fft, ifft
    registerFFTKernels(registry);

    // Shape operations (Kernel mechanism)
    registerShapeKernels(registry);

    // Audio operations - includes iir.biquad, etc.
    registerAudioKernels(registry);
}
