/**
 * v5 Internal: square
 * Mechanism: Iterator
 * DispatchKey: square
 *
 * 逐元素平方: self * self
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.square;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function square(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'square',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
