/**
 * v5 Internal: reshape
 * Mechanism: View
 * DispatchKey: reshape
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.reshape;
import { dispatchView, type OperatorContext } from '../../dispatch/handlers';

export function reshape(
    self: ITensorHandle,
    shape: readonly number[]
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'reshape',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: { shape },
    };
    return dispatchView(__entry, ctx);
}
