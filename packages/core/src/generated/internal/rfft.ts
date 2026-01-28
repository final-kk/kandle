/**
 * v5 Internal: rfft
 * Mechanism: FFT
 * DispatchKey: rfft
 *
 * Computes the one dimensional FFT of real-valued input.
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.rfft;
import { dispatchFFT, type OperatorContext } from '../../dispatch/handlers';

export function rfft(
    input: ITensorHandle,
    n?: number | undefined,
    dim?: number,
    norm?: 'forward' | 'backward' | 'ortho' | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'rfft',
        tensorInputs: [input],
        scalarArgs: { n, dim, norm } as Record<string, any>,
        metadata: { n, dim, norm },
    };
    return dispatchFFT(__entry, ctx) as ITensorHandle;
}
