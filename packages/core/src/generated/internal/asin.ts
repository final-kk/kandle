/**
 * v5 Internal: asin
 * Mechanism: Iterator
 * DispatchKey: asin
 *
 * 逐元素反正弦: arcsin(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.asin;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function asin(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'asin',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
