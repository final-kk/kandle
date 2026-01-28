/**
 * v5 Internal: ne.Tensor
 * Mechanism: Iterator
 * DispatchKey: ne
 *
 * 逐元素不等比较: self != other
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.ne_Tensor;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function ne_Tensor(
    self: ITensorHandle,
    other: ITensorHandle
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'ne',
        tensorInputs: [self, other],
        scalarArgs: {},
        metadata: {},
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
