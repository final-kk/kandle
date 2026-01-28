/**
 * v5 Internal: ge.Scalar
 * Mechanism: Iterator
 * DispatchKey: ge_scalar
 *
 * 逐元素大于等于比较: self >= other (标量版本)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.ge_Scalar;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function ge_Scalar(
    self: ITensorHandle,
    other: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'ge_scalar',
        tensorInputs: [self],
        scalarArgs: { other } as Record<string, any>,
        metadata: { other },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
