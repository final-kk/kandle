/**
 * v5 Internal: clone
 * Mechanism: Copy
 * DispatchKey: clone
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.clone;
import { dispatchCopy, type OperatorContext } from '../../dispatch/handlers';

export function clone(
    self: ITensorHandle
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'clone',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
    };
    return dispatchCopy(__entry, ctx);
}
