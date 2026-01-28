/**
 * v5 Internal: cos
 * Mechanism: Iterator
 * DispatchKey: cos
 *
 * 逐元素余弦: cos(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.cos;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function cos(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'cos',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
