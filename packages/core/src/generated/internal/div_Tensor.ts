/**
 * v5 Internal: div.Tensor
 * Mechanism: Iterator
 * DispatchKey: div
 *
 * 逐元素除法: self / other
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.div_Tensor;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function div_Tensor(
    self: ITensorHandle,
    other: ITensorHandle,
    roundingMode?: 'trunc' | 'floor' | undefined,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'div',
        tensorInputs: [self, other],
        scalarArgs: { roundingMode } as Record<string, any>,
        metadata: { roundingMode },
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
