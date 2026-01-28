/**
 * v5 Internal: neg
 * Mechanism: Iterator
 * DispatchKey: neg
 *
 * 逐元素取反: -self
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.neg;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function neg(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'neg',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
