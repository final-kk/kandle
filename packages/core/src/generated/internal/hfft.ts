/**
 * v5 Internal: hfft
 * Mechanism: Composite
 * DispatchKey: hfft
 *
 * Computes the 1D discrete Fourier transform of a Hermitian symmetric input signal.
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.hfft;
import { dispatchComposite, type OperatorContext } from '../../dispatch/handlers';

export function hfft(
    input: ITensorHandle,
    n?: number | undefined,
    dim?: number,
    norm?: 'forward' | 'backward' | 'ortho' | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'hfft',
        tensorInputs: [input],
        scalarArgs: { n, dim, norm } as Record<string, any>,
        metadata: { n, dim, norm },
    };
    return dispatchComposite(__entry, ctx) as ITensorHandle;
}
