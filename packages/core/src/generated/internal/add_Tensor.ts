/**
 * v5 Internal: add.Tensor
 * Mechanism: Iterator
 * DispatchKey: add
 *
 * 逐元素加法: self + alpha * other
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.add_Tensor;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function add_Tensor(
    self: ITensorHandle,
    other: ITensorHandle,
    alpha?: number,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'add',
        tensorInputs: [self, other],
        scalarArgs: { alpha } as Record<string, any>,
        metadata: { alpha },
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
