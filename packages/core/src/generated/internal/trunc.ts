/**
 * v5 Internal: trunc
 * Mechanism: Iterator
 * DispatchKey: trunc
 *
 * 逐元素向零取整: trunc(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.trunc;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function trunc(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'trunc',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
