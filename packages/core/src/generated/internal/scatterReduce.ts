/**
 * v5 Internal: scatterReduce
 * Mechanism: Scatter
 * DispatchKey: scatter_reduce
 *
 * 通用散射归约
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.scatterReduce;
import { dispatchScatter, type OperatorContext } from '../../dispatch/handlers';

export function scatterReduce(
    self: ITensorHandle,
    dim: number,
    index: ITensorHandle,
    src: ITensorHandle,
    reduce: 'sum' | 'prod' | 'mean' | 'amax' | 'amin',
    includeSelf?: boolean,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'scatter_reduce',
        tensorInputs: [self, index, src],
        scalarArgs: { reduce, includeSelf } as Record<string, any>,
        metadata: { dim, reduce, includeSelf },
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchScatter(__entry, ctx) as ITensorHandle;
}
