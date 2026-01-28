/**
 * v5 Internal: unsqueeze
 * Mechanism: View
 * DispatchKey: unsqueeze
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.unsqueeze;
import { dispatchView, type OperatorContext } from '../../dispatch/handlers';

export function unsqueeze(
    self: ITensorHandle,
    dim: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'unsqueeze',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: { dim },
    };
    return dispatchView(__entry, ctx);
}
