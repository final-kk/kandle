/**
 * v5 Internal: maximum
 * Mechanism: Iterator
 * DispatchKey: maximum
 *
 * 逐元素最大值: max(self, other)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.maximum;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function maximum(
    self: ITensorHandle,
    other: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'maximum',
        tensorInputs: [self, other],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
