/**
 * v5 Internal: le.Scalar
 * Mechanism: Iterator
 * DispatchKey: le_scalar
 *
 * 逐元素小于等于比较: self <= other (标量版本)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.le_Scalar;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function le_Scalar(
    self: ITensorHandle,
    other: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'le_scalar',
        tensorInputs: [self],
        scalarArgs: { other } as Record<string, any>,
        metadata: { other },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
