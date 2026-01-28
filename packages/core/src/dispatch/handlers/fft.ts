/**
 * FFT Handler
 *
 * 处理 FFT 操作: fft, ifft, rfft, irfft
 * 
 * Mechanism: Factory (creates new output tensor)
 * 
 * The kernel now supports FFT on arbitrary dimension via strided access.
 */

import type { ITensorHandle, DType, DeviceNameEnum } from '@kandle/types';
import type { OpEntry } from '@kandle/types';
import { env } from '../../env';
import type { PatternHandler, OperatorContext, DirectContext } from './types';
import { isComplexDtype } from '@kandle/utils';

// FFT types (inline to avoid cross-package import issues)
type FFTNorm = 'forward' | 'backward' | 'ortho';
type FFTDirection = 'forward' | 'inverse';

interface FFTKernelArgs {
    input: ITensorHandle;
    output: ITensorHandle;
    dim: number;
    n: number;
    norm: FFTNorm;
    direction: FFTDirection;
    isRealInput: boolean;
}

/**
 * FFT Handler Class
 */
export class FFTHandler implements PatternHandler {
    private static instance: FFTHandler;

    static getInstance(): FFTHandler {
        if (!FFTHandler.instance) {
            FFTHandler.instance = new FFTHandler();
        }
        return FFTHandler.instance;
    }

    buildContext(entry: OpEntry, ctx: OperatorContext): DirectContext {
        return {
            kind: 'direct',
            inputs: ctx.tensorInputs,
            scalars: ctx.scalarArgs,
            metadata: ctx.metadata,
            outs: ctx.outs,
            kernelName: entry.dispatchKey,
        };
    }

    execute(execCtx: DirectContext): ITensorHandle {
        const { inputs, scalars, metadata, kernelName } = execCtx;
        const allParams = { ...scalars, ...metadata };
        const device = (allParams['device'] as DeviceNameEnum) ?? env.getDefaultDevice().name;
        const backend = env.getBackend(device);

        // Get input tensor
        const input = inputs[0];
        if (!input) {
            throw new Error('FFT requires an input tensor');
        }

        // Get parameters
        const dim = (allParams['dim'] ?? -1) as number;
        const norm = (allParams['norm'] ?? 'backward') as FFTNorm;
        const n = allParams['n'] as number | undefined;

        // Resolve dimension
        const inputShape = input.shape;
        const ndim = inputShape.length;
        const resolvedDim = dim < 0 ? ndim + dim : dim;

        if (resolvedDim < 0 || resolvedDim >= ndim) {
            throw new Error(`Invalid dimension ${dim} for tensor with ${ndim} dimensions`);
        }

        // Determine FFT size
        const fftSize = n ?? inputShape[resolvedDim];

        // Note: Power-of-2 sizes use Cooley-Tukey radix-2 algorithm.
        // Non-power-of-2 sizes use Bluestein algorithm (Chirp Z-Transform).
        // Both are handled in the executor layer.

        // Determine direction and output dtype
        let direction: FFTDirection = 'forward';
        let outputDtype: DType = 'complex64';

        switch (kernelName) {
            case 'fft':
                direction = 'forward';
                outputDtype = 'complex64';
                break;
            case 'ifft':
                direction = 'inverse';
                outputDtype = 'complex64';
                break;
            case 'rfft':
                direction = 'forward';
                outputDtype = 'complex64';
                break;
            case 'irfft':
                direction = 'inverse';
                outputDtype = 'float32';
                break;
        }

        // Compute output shape
        const outputShape = [...inputShape];
        if (kernelName === 'rfft') {
            outputShape[resolvedDim] = Math.floor(fftSize / 2) + 1;
        } else if (kernelName === 'irfft') {
            outputShape[resolvedDim] = n ?? (inputShape[resolvedDim] - 1) * 2;
        } else {
            outputShape[resolvedDim] = fftSize;
        }

        // Create output tensor
        const output = backend.createTensorHandle(outputShape, outputDtype);

        // Check if input is real or complex
        const isRealInput = !isComplexDtype(input.dtype);

        // Build kernel args and dispatch
        switch (kernelName) {
            case 'fft':
            case 'ifft': {
                const args: FFTKernelArgs = {
                    input,
                    output,
                    dim: resolvedDim,
                    n: fftSize,
                    norm,
                    direction,
                    isRealInput,
                };

                const kernel = backend.operators.find(kernelName);
                if (!kernel) {
                    throw new Error(`${kernelName} kernel not available on backend`);
                }

                const ctx: DirectContext = {
                    kind: 'direct',
                    inputs: [input],
                    scalars: args as any,
                    metadata: {},
                    outs: [output],
                    kernelName,
                };

                (kernel as (ctx: DirectContext) => void)(ctx);
                break;
            }

            case 'rfft': {
                const onesidedLen = Math.floor(fftSize / 2) + 1;
                const args = {
                    input,
                    output,
                    dim: resolvedDim,
                    n: fftSize,
                    norm,
                    onesidedLen,
                };

                const kernel = backend.operators.find('rfft');
                if (!kernel) {
                    throw new Error('rfft kernel not available on backend');
                }

                const ctx: DirectContext = {
                    kind: 'direct',
                    inputs: [input],
                    scalars: args as any,
                    metadata: {},
                    outs: [output],
                    kernelName: 'rfft',
                };

                (kernel as (ctx: DirectContext) => void)(ctx);
                break;
            }

            case 'irfft': {
                const inputOnesidedLen = inputShape[resolvedDim];
                const outputLen = n ?? (inputOnesidedLen - 1) * 2;
                const args = {
                    input,
                    output,
                    dim: resolvedDim,
                    n: outputLen,
                    norm,
                    onesidedLen: inputOnesidedLen,
                };

                const kernel = backend.operators.find('irfft');
                if (!kernel) {
                    throw new Error('irfft kernel not available on backend');
                }

                const ctx: DirectContext = {
                    kind: 'direct',
                    inputs: [input],
                    scalars: args as any,
                    metadata: {},
                    outs: [output],
                    kernelName: 'irfft',
                };

                (kernel as (ctx: DirectContext) => void)(ctx);
                break;
            }

            default:
                throw new Error(`Unknown FFT operation: ${kernelName}`);
        }

        return output;
    }

    static dispatch(entry: OpEntry, ctx: OperatorContext): ITensorHandle {
        const handler = FFTHandler.getInstance();
        const execCtx = handler.buildContext(entry, ctx);
        return handler.execute(execCtx);
    }
}

export const dispatchFFT = FFTHandler.dispatch;

