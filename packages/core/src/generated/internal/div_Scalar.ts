/**
 * v5 Internal: div.Scalar
 * Mechanism: Iterator
 * DispatchKey: div_scalar
 *
 * 逐元素除法: self / other (标量版本)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.div_Scalar;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function div_Scalar(
    self: ITensorHandle,
    other: number,
    roundingMode?: 'trunc' | 'floor' | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'div_scalar',
        tensorInputs: [self],
        scalarArgs: { other, roundingMode } as Record<string, any>,
        metadata: { other, roundingMode },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
