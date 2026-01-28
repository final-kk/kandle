/**
 * v5 Internal: eq.Scalar
 * Mechanism: Iterator
 * DispatchKey: eq_scalar
 *
 * 逐元素相等比较: self == other (标量版本)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.eq_Scalar;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function eq_Scalar(
    self: ITensorHandle,
    other: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'eq_scalar',
        tensorInputs: [self],
        scalarArgs: { other } as Record<string, any>,
        metadata: { other },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
