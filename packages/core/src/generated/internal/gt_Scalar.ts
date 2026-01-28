/**
 * v5 Internal: gt.Scalar
 * Mechanism: Iterator
 * DispatchKey: gt_scalar
 *
 * 逐元素大于比较: self > other (标量版本)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.gt_Scalar;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function gt_Scalar(
    self: ITensorHandle,
    other: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'gt_scalar',
        tensorInputs: [self],
        scalarArgs: { other } as Record<string, any>,
        metadata: { other },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
