/**
 * v5 Internal: ifft
 * Mechanism: FFT
 * DispatchKey: ifft
 *
 * Computes the one dimensional inverse discrete Fourier transform of input.
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.ifft;
import { dispatchFFT, type OperatorContext } from '../../dispatch/handlers';

export function ifft(
    input: ITensorHandle,
    n?: number | undefined,
    dim?: number,
    norm?: 'forward' | 'backward' | 'ortho' | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'ifft',
        tensorInputs: [input],
        scalarArgs: { n, dim, norm } as Record<string, any>,
        metadata: { n, dim, norm },
    };
    return dispatchFFT(__entry, ctx) as ITensorHandle;
}
