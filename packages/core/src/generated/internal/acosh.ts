/**
 * v5 Internal: acosh
 * Mechanism: Iterator
 * DispatchKey: acosh
 *
 * 逐元素反双曲余弦: arccosh(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.acosh;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function acosh(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'acosh',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
