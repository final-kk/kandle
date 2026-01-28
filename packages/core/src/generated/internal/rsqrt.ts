/**
 * v5 Internal: rsqrt
 * Mechanism: Iterator
 * DispatchKey: rsqrt
 *
 * 逐元素平方根倒数: 1/sqrt(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.rsqrt;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function rsqrt(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'rsqrt',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
