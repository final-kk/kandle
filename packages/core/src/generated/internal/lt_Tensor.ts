/**
 * v5 Internal: lt.Tensor
 * Mechanism: Iterator
 * DispatchKey: lt
 *
 * 逐元素小于比较: self < other
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.lt_Tensor;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function lt_Tensor(
    self: ITensorHandle,
    other: ITensorHandle
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'lt',
        tensorInputs: [self, other],
        scalarArgs: {},
        metadata: {},
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
