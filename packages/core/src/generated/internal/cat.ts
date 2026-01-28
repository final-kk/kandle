/**
 * v5 Internal: cat
 * Mechanism: Shape
 * DispatchKey: cat
 *
 * 沿指定维度拼接张量序列
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.cat;
import { dispatchShape, type OperatorContext } from '../../dispatch/handlers';

export function cat(
    tensors: ITensorHandle[],
    dim?: number,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'cat',
        tensorInputs: [],
        scalarArgs: {},
        metadata: { tensors, dim },
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchShape(__entry, ctx) as ITensorHandle;
}
