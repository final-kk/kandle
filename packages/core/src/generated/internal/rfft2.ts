/**
 * v5 Internal: rfft2
 * Mechanism: Composite
 * DispatchKey: rfft2
 *
 * Computes the 2-dimensional FFT of real-valued input.
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.rfft2;
import { dispatchComposite, type OperatorContext } from '../../dispatch/handlers';

export function rfft2(
    input: ITensorHandle,
    s?: readonly number[] | undefined,
    dim?: number | readonly number[] | undefined,
    norm?: 'forward' | 'backward' | 'ortho' | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'rfft2',
        tensorInputs: [input],
        scalarArgs: { norm } as Record<string, any>,
        metadata: { s, dim, norm },
    };
    return dispatchComposite(__entry, ctx) as ITensorHandle;
}
