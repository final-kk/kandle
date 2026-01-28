/**
 * v5 Internal: stack
 * Mechanism: Shape
 * DispatchKey: stack
 *
 * 沿新维度堆叠张量序列
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.stack;
import { dispatchShape, type OperatorContext } from '../../dispatch/handlers';

export function stack(
    tensors: ITensorHandle[],
    dim?: number,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'stack',
        tensorInputs: [],
        scalarArgs: {},
        metadata: { tensors, dim },
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchShape(__entry, ctx) as ITensorHandle;
}
