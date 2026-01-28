/**
 * v5 Internal: atan
 * Mechanism: Iterator
 * DispatchKey: atan
 *
 * 逐元素反正切: arctan(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.atan;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function atan(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'atan',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
