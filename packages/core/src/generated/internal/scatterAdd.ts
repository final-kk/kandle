/**
 * v5 Internal: scatterAdd
 * Mechanism: Scatter
 * DispatchKey: scatter_add
 *
 * 散射加法: out[index[...]][...] += src[...]
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.scatterAdd;
import { dispatchScatter, type OperatorContext } from '../../dispatch/handlers';

export function scatterAdd(
    self: ITensorHandle,
    dim: number,
    index: ITensorHandle,
    src: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'scatter_add',
        tensorInputs: [self, index, src],
        scalarArgs: {},
        metadata: { dim },
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchScatter(__entry, ctx) as ITensorHandle;
}
