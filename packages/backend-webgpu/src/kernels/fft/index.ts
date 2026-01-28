/**
 * FFT Kernel Module
 * 
 * Exports FFT types, executors, and registration.
 */

export * from './types';
export * from './executor';

import type { IBackendOpsRegister, DirectContext } from '@kandle/types';
import { executeFFT, executeRFFT, executeIRFFT } from './executor';
import type { FFTKernelArgs, RFFTKernelArgs, IRFFTKernelArgs } from './types';

/**
 * Register FFT kernels
 */
export function registerFFTKernels(registry: IBackendOpsRegister): void {
    // FFT: forward complex-to-complex
    registry.register('fft', (ctx: DirectContext) => {
        const args: FFTKernelArgs = ctx.scalars as any;
        executeFFT(args);
    });

    // IFFT: inverse complex-to-complex (uses same executor with direction=inverse)
    registry.register('ifft', (ctx: DirectContext) => {
        const args: FFTKernelArgs = ctx.scalars as any;
        executeFFT(args);
    });

    // RFFT: real-to-complex onesided
    registry.register('rfft', (ctx: DirectContext) => {
        const args: RFFTKernelArgs = ctx.scalars as any;
        executeRFFT(args);
    });

    // IRFFT: complex onesided to real
    registry.register('irfft', (ctx: DirectContext) => {
        const args: IRFFTKernelArgs = ctx.scalars as any;
        executeIRFFT(args);
    });
}

