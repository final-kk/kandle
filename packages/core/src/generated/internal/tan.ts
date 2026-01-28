/**
 * v5 Internal: tan
 * Mechanism: Iterator
 * DispatchKey: tan
 *
 * 逐元素正切: tan(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.tan;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function tan(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'tan',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
