/**
 * v5 Internal: transpose
 * Mechanism: View
 * DispatchKey: transpose
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.transpose;
import { dispatchView, type OperatorContext } from '../../dispatch/handlers';

export function transpose(
    self: ITensorHandle,
    dim0: number,
    dim1: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'transpose',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: { dim0, dim1 },
    };
    return dispatchView(__entry, ctx);
}
