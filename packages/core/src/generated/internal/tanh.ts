/**
 * v5 Internal: tanh
 * Mechanism: Iterator
 * DispatchKey: tanh
 *
 * 逐元素双曲正切: tanh(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.tanh;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function tanh(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'tanh',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
