/**
 * v5 Internal: ceil
 * Mechanism: Iterator
 * DispatchKey: ceil
 *
 * 逐元素向上取整: ceil(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.ceil;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function ceil(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'ceil',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
