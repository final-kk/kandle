/**
 * v5 Internal: fftn
 * Mechanism: Composite
 * DispatchKey: fftn
 *
 * Computes the N-dimensional discrete Fourier transform of input.
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.fftn;
import { dispatchComposite, type OperatorContext } from '../../dispatch/handlers';

export function fftn(
    input: ITensorHandle,
    s?: readonly number[] | undefined,
    dim?: number | readonly number[] | undefined,
    norm?: 'forward' | 'backward' | 'ortho' | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'fftn',
        tensorInputs: [input],
        scalarArgs: { norm } as Record<string, any>,
        metadata: { s, dim, norm },
    };
    return dispatchComposite(__entry, ctx) as ITensorHandle;
}
