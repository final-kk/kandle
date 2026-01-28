/**
 * v5 Internal: eq.Tensor
 * Mechanism: Iterator
 * DispatchKey: eq
 *
 * 逐元素相等比较: self == other
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.eq_Tensor;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function eq_Tensor(
    self: ITensorHandle,
    other: ITensorHandle
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'eq',
        tensorInputs: [self, other],
        scalarArgs: {},
        metadata: {},
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
