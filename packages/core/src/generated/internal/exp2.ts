/**
 * v5 Internal: exp2
 * Mechanism: Iterator
 * DispatchKey: exp2
 *
 * 逐元素 2 的幂: 2^self
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.exp2;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function exp2(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'exp2',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
