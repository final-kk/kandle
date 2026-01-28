/**
 * v5 Internal: floor
 * Mechanism: Iterator
 * DispatchKey: floor
 *
 * 逐元素向下取整: floor(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.floor;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function floor(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'floor',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
