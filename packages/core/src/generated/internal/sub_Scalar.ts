/**
 * v5 Internal: sub.Scalar
 * Mechanism: Iterator
 * DispatchKey: sub_scalar
 *
 * 逐元素减法: self - alpha * other (标量版本)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.sub_Scalar;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function sub_Scalar(
    self: ITensorHandle,
    other: number,
    alpha?: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'sub_scalar',
        tensorInputs: [self],
        scalarArgs: { other, alpha } as Record<string, any>,
        metadata: { other, alpha },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
