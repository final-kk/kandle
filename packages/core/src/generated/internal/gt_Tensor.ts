/**
 * v5 Internal: gt.Tensor
 * Mechanism: Iterator
 * DispatchKey: gt
 *
 * 逐元素大于比较: self > other
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.gt_Tensor;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function gt_Tensor(
    self: ITensorHandle,
    other: ITensorHandle
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'gt',
        tensorInputs: [self, other],
        scalarArgs: {},
        metadata: {},
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
