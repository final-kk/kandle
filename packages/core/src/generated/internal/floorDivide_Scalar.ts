/**
 * v5 Internal: floorDivide.Scalar
 * Mechanism: Iterator
 * DispatchKey: floor_divide_scalar
 *
 * 向下取整除法: floor(self / other) (标量版本)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.floorDivide_Scalar;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function floorDivide_Scalar(
    self: ITensorHandle,
    other: number,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'floor_divide_scalar',
        tensorInputs: [self],
        scalarArgs: { other } as Record<string, any>,
        metadata: { other },
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
