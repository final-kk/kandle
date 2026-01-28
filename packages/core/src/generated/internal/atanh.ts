/**
 * v5 Internal: atanh
 * Mechanism: Iterator
 * DispatchKey: atanh
 *
 * 逐元素反双曲正切: arctanh(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.atanh;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function atanh(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'atanh',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
