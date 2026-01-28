/**
 * v5 Internal: sin
 * Mechanism: Iterator
 * DispatchKey: sin
 *
 * 逐元素正弦: sin(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.sin;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function sin(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'sin',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
