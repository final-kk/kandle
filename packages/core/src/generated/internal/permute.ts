/**
 * v5 Internal: permute
 * Mechanism: View
 * DispatchKey: permute
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.permute;
import { dispatchView, type OperatorContext } from '../../dispatch/handlers';

export function permute(
    self: ITensorHandle,
    dims: number | readonly number[]
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'permute',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: { dims },
    };
    return dispatchView(__entry, ctx);
}
