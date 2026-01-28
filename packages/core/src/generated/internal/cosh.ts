/**
 * v5 Internal: cosh
 * Mechanism: Iterator
 * DispatchKey: cosh
 *
 * 逐元素双曲余弦: cosh(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.cosh;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function cosh(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'cosh',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
