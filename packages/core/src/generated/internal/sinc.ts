/**
 * v5 Internal: sinc
 * Mechanism: Iterator
 * DispatchKey: sinc
 *
 * 逐元素归一化 sinc: sin(πx)/(πx)，x=0 时返回 1
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.sinc;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function sinc(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'sinc',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
