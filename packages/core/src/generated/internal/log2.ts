/**
 * v5 Internal: log2
 * Mechanism: Iterator
 * DispatchKey: log2
 *
 * 逐元素以 2 为底对数: log2(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.log2;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function log2(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'log2',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
