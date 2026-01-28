/**
 * v5 Internal: log
 * Mechanism: Iterator
 * DispatchKey: log
 *
 * 逐元素自然对数: ln(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.log;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function log(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'log',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
