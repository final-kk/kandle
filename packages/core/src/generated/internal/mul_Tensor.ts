/**
 * v5 Internal: mul.Tensor
 * Mechanism: Iterator
 * DispatchKey: mul
 *
 * 逐元素乘法: self * other
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.mul_Tensor;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function mul_Tensor(
    self: ITensorHandle,
    other: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'mul',
        tensorInputs: [self, other],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
