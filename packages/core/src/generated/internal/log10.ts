/**
 * v5 Internal: log10
 * Mechanism: Iterator
 * DispatchKey: log10
 *
 * 逐元素以 10 为底对数: log10(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.log10;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function log10(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'log10',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
