/**
 * v5 Internal: sub.Tensor
 * Mechanism: Iterator
 * DispatchKey: sub
 *
 * 逐元素减法: self - alpha * other
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.sub_Tensor;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function sub_Tensor(
    self: ITensorHandle,
    other: ITensorHandle,
    alpha?: number,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'sub',
        tensorInputs: [self, other],
        scalarArgs: { alpha } as Record<string, any>,
        metadata: { alpha },
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
