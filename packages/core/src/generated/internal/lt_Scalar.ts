/**
 * v5 Internal: lt.Scalar
 * Mechanism: Iterator
 * DispatchKey: lt_scalar
 *
 * 逐元素小于比较: self < other (标量版本)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.lt_Scalar;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function lt_Scalar(
    self: ITensorHandle,
    other: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'lt_scalar',
        tensorInputs: [self],
        scalarArgs: { other } as Record<string, any>,
        metadata: { other },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
