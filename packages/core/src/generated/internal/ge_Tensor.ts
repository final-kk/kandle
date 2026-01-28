/**
 * v5 Internal: ge.Tensor
 * Mechanism: Iterator
 * DispatchKey: ge
 *
 * 逐元素大于等于比较: self >= other
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.ge_Tensor;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function ge_Tensor(
    self: ITensorHandle,
    other: ITensorHandle
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'ge',
        tensorInputs: [self, other],
        scalarArgs: {},
        metadata: {},
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
