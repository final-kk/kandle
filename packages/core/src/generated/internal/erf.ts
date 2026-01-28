/**
 * v5 Internal: erf
 * Mechanism: Iterator
 * DispatchKey: erf
 *
 * 逐元素误差函数: erf(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.erf;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function erf(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'erf',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
