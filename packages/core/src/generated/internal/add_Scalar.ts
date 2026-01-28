/**
 * v5 Internal: add.Scalar
 * Mechanism: Iterator
 * DispatchKey: add_scalar
 *
 * 逐元素加法: self + alpha * other (标量版本)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.add_Scalar;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function add_Scalar(
    self: ITensorHandle,
    other: number,
    alpha?: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'add_scalar',
        tensorInputs: [self],
        scalarArgs: { other, alpha } as Record<string, any>,
        metadata: { other, alpha },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
