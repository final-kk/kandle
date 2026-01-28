/**
 * v5 Internal: mul.Scalar
 * Mechanism: Iterator
 * DispatchKey: mul_scalar
 *
 * 逐元素乘法: self * other (标量版本)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.mul_Scalar;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function mul_Scalar(
    self: ITensorHandle,
    other: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'mul_scalar',
        tensorInputs: [self],
        scalarArgs: { other } as Record<string, any>,
        metadata: { other },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
