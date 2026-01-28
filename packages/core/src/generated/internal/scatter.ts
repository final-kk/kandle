/**
 * v5 Internal: scatter
 * Mechanism: Scatter
 * DispatchKey: scatter
 *
 * 散射操作: out[index[...]][...] = src[...]
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.scatter;
import { dispatchScatter, type OperatorContext } from '../../dispatch/handlers';

export function scatter(
    self: ITensorHandle,
    dim: number,
    index: ITensorHandle,
    src: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'scatter',
        tensorInputs: [self, index, src],
        scalarArgs: {},
        metadata: { dim },
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchScatter(__entry, ctx) as ITensorHandle;
}
