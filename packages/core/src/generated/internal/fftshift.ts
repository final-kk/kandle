/**
 * v5 Internal: fftshift
 * Mechanism: Composite
 * DispatchKey: fftshift
 *
 * Shift zero-frequency component to center of spectrum.
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.fftshift;
import { dispatchComposite, type OperatorContext } from '../../dispatch/handlers';

export function fftshift(
    input: ITensorHandle,
    dim?: number | readonly number[] | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'fftshift',
        tensorInputs: [input],
        scalarArgs: {},
        metadata: { dim },
    };
    return dispatchComposite(__entry, ctx) as ITensorHandle;
}
