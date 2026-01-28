/**
 * v5 Internal: minimum
 * Mechanism: Iterator
 * DispatchKey: minimum
 *
 * 逐元素最小值: min(self, other)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.minimum;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function minimum(
    self: ITensorHandle,
    other: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'minimum',
        tensorInputs: [self, other],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
