/**
 * v5 Internal: flatten
 * Mechanism: View
 * DispatchKey: flatten
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.flatten;
import { dispatchView, type OperatorContext } from '../../dispatch/handlers';

export function flatten(
    self: ITensorHandle,
    startDim?: number,
    endDim?: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'flatten',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: { startDim, endDim },
    };
    return dispatchView(__entry, ctx);
}
