/**
 * v5 Internal: i0
 * Mechanism: Iterator
 * DispatchKey: i0
 *
 * 逐元素计算 0 阶修正贝塞尔函数: I₀(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.i0;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function i0(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'i0',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
