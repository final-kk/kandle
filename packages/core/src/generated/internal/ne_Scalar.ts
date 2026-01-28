/**
 * v5 Internal: ne.Scalar
 * Mechanism: Iterator
 * DispatchKey: ne_scalar
 *
 * 逐元素不等比较: self != other (标量版本)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.ne_Scalar;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function ne_Scalar(
    self: ITensorHandle,
    other: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'ne_scalar',
        tensorInputs: [self],
        scalarArgs: { other } as Record<string, any>,
        metadata: { other },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
