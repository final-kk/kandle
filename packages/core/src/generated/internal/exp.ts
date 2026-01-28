/**
 * v5 Internal: exp
 * Mechanism: Iterator
 * DispatchKey: exp
 *
 * 逐元素指数: e^self
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.exp;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function exp(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'exp',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
