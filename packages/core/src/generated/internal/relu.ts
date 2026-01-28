/**
 * v5 Internal: relu
 * Mechanism: Iterator
 * DispatchKey: relu
 *
 * 逐元素 ReLU: max(0, self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.relu;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function relu(
    self: ITensorHandle
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'relu',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
