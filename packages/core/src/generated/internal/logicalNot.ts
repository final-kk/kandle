/**
 * v5 Internal: logicalNot
 * Mechanism: Iterator
 * DispatchKey: logical_not
 *
 * 逐元素逻辑非: !self
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.logicalNot;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function logicalNot(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'logical_not',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
