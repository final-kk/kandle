/**
 * v5 Internal: indexSelect
 * Mechanism: Gather
 * DispatchKey: index_select
 *
 * 沿维度选择索引: out[...] = self.select(dim, index[...])
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.indexSelect;
import { dispatchGather, type OperatorContext } from '../../dispatch/handlers';

export function indexSelect(
    self: ITensorHandle,
    dim: number,
    index: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'index_select',
        tensorInputs: [self, index],
        scalarArgs: {},
        metadata: { dim },
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchGather(__entry, ctx) as ITensorHandle;
}
