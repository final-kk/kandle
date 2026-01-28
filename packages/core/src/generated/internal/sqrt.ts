/**
 * v5 Internal: sqrt
 * Mechanism: Iterator
 * DispatchKey: sqrt
 *
 * 逐元素平方根: sqrt(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.sqrt;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function sqrt(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'sqrt',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
