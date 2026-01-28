/**
 * v5 Internal: le.Tensor
 * Mechanism: Iterator
 * DispatchKey: le
 *
 * 逐元素小于等于比较: self <= other
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.le_Tensor;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function le_Tensor(
    self: ITensorHandle,
    other: ITensorHandle
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'le',
        tensorInputs: [self, other],
        scalarArgs: {},
        metadata: {},
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
