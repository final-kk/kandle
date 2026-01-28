/**
 * v5 Internal: ifftshift
 * Mechanism: Composite
 * DispatchKey: ifftshift
 *
 * Inverse of fftshift.
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.ifftshift;
import { dispatchComposite, type OperatorContext } from '../../dispatch/handlers';

export function ifftshift(
    input: ITensorHandle,
    dim?: number | readonly number[] | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'ifftshift',
        tensorInputs: [input],
        scalarArgs: {},
        metadata: { dim },
    };
    return dispatchComposite(__entry, ctx) as ITensorHandle;
}
