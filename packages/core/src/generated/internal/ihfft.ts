/**
 * v5 Internal: ihfft
 * Mechanism: Composite
 * DispatchKey: ihfft
 *
 * Computes the inverse of hfft.
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.ihfft;
import { dispatchComposite, type OperatorContext } from '../../dispatch/handlers';

export function ihfft(
    input: ITensorHandle,
    n?: number | undefined,
    dim?: number,
    norm?: 'forward' | 'backward' | 'ortho' | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'ihfft',
        tensorInputs: [input],
        scalarArgs: { n, dim, norm } as Record<string, any>,
        metadata: { n, dim, norm },
    };
    return dispatchComposite(__entry, ctx) as ITensorHandle;
}
