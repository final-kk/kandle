/**
 * v5 Internal: irfftn
 * Mechanism: Composite
 * DispatchKey: irfftn
 *
 * Computes the N-dimensional inverse FFT of real input.
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.irfftn;
import { dispatchComposite, type OperatorContext } from '../../dispatch/handlers';

export function irfftn(
    input: ITensorHandle,
    s?: readonly number[] | undefined,
    dim?: number | readonly number[] | undefined,
    norm?: 'forward' | 'backward' | 'ortho' | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'irfftn',
        tensorInputs: [input],
        scalarArgs: { norm } as Record<string, any>,
        metadata: { s, dim, norm },
    };
    return dispatchComposite(__entry, ctx) as ITensorHandle;
}
