/**
 * v5 Internal: expand
 * Mechanism: View
 * DispatchKey: expand
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.expand;
import { dispatchView, type OperatorContext } from '../../dispatch/handlers';

export function expand(
    self: ITensorHandle,
    size: readonly number[]
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'expand',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: { size },
    };
    return dispatchView(__entry, ctx);
}
