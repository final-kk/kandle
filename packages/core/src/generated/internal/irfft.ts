/**
 * v5 Internal: irfft
 * Mechanism: FFT
 * DispatchKey: irfft
 *
 * Computes the inverse FFT of rfft. Output is real-valued.
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.irfft;
import { dispatchFFT, type OperatorContext } from '../../dispatch/handlers';

export function irfft(
    input: ITensorHandle,
    n?: number | undefined,
    dim?: number,
    norm?: 'forward' | 'backward' | 'ortho' | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'irfft',
        tensorInputs: [input],
        scalarArgs: { n, dim, norm } as Record<string, any>,
        metadata: { n, dim, norm },
    };
    return dispatchFFT(__entry, ctx) as ITensorHandle;
}
