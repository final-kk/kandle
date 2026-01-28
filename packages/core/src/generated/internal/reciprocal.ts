/**
 * v5 Internal: reciprocal
 * Mechanism: Iterator
 * DispatchKey: reciprocal
 *
 * 逐元素倒数: 1/self
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.reciprocal;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function reciprocal(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'reciprocal',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
