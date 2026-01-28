/**
 * v5 Internal: rfftn
 * Mechanism: Composite
 * DispatchKey: rfftn
 *
 * Computes the N-dimensional FFT of real-valued input.
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.rfftn;
import { dispatchComposite, type OperatorContext } from '../../dispatch/handlers';

export function rfftn(
    input: ITensorHandle,
    s?: readonly number[] | undefined,
    dim?: number | readonly number[] | undefined,
    norm?: 'forward' | 'backward' | 'ortho' | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'rfftn',
        tensorInputs: [input],
        scalarArgs: { norm } as Record<string, any>,
        metadata: { s, dim, norm },
    };
    return dispatchComposite(__entry, ctx) as ITensorHandle;
}
