/**
 * v5 Internal: fft
 * Mechanism: FFT
 * DispatchKey: fft
 *
 * Computes the one dimensional discrete Fourier transform of input.
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.fft;
import { dispatchFFT, type OperatorContext } from '../../dispatch/handlers';

export function fft(
    input: ITensorHandle,
    n?: number | undefined,
    dim?: number,
    norm?: 'forward' | 'backward' | 'ortho' | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'fft',
        tensorInputs: [input],
        scalarArgs: { n, dim, norm } as Record<string, any>,
        metadata: { n, dim, norm },
    };
    return dispatchFFT(__entry, ctx) as ITensorHandle;
}
