/**
 * v5 Internal: sinh
 * Mechanism: Iterator
 * DispatchKey: sinh
 *
 * 逐元素双曲正弦: sinh(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.sinh;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function sinh(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'sinh',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
