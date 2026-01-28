/**
 * v5 Internal: irfft2
 * Mechanism: Composite
 * DispatchKey: irfft2
 *
 * Computes the inverse 2D FFT of rfft2 output. Output is real-valued.
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.irfft2;
import { dispatchComposite, type OperatorContext } from '../../dispatch/handlers';

export function irfft2(
    input: ITensorHandle,
    s?: readonly number[] | undefined,
    dim?: number | readonly number[] | undefined,
    norm?: 'forward' | 'backward' | 'ortho' | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'irfft2',
        tensorInputs: [input],
        scalarArgs: { norm } as Record<string, any>,
        metadata: { s, dim, norm },
    };
    return dispatchComposite(__entry, ctx) as ITensorHandle;
}
