/**
 * v5 Internal: acos
 * Mechanism: Iterator
 * DispatchKey: acos
 *
 * 逐元素反余弦: arccos(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.acos;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function acos(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'acos',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
