/**
 * v5 Internal: floorDivide.Tensor
 * Mechanism: Iterator
 * DispatchKey: floor_divide
 *
 * 向下取整除法: floor(self / other)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.floorDivide_Tensor;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function floorDivide_Tensor(
    self: ITensorHandle,
    other: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'floor_divide',
        tensorInputs: [self, other],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
