/**
 * v5 Internal: fft2
 * Mechanism: Composite
 * DispatchKey: fft2
 *
 * Computes the 2-dimensional discrete Fourier transform of input.
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.fft2;
import { dispatchComposite, type OperatorContext } from '../../dispatch/handlers';

export function fft2(
    input: ITensorHandle,
    s?: readonly number[] | undefined,
    dim?: number | readonly number[] | undefined,
    norm?: 'forward' | 'backward' | 'ortho' | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'fft2',
        tensorInputs: [input],
        scalarArgs: { norm } as Record<string, any>,
        metadata: { s, dim, norm },
    };
    return dispatchComposite(__entry, ctx) as ITensorHandle;
}
