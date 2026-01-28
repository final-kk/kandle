/**
 * v5 Internal: where
 * Mechanism: Iterator
 * DispatchKey: where
 *
 * 根据条件选择: condition ? self : other
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.where;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function where(
    condition: ITensorHandle,
    self: ITensorHandle,
    other: ITensorHandle
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'where',
        tensorInputs: [condition, self, other],
        scalarArgs: {},
        metadata: {},
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
