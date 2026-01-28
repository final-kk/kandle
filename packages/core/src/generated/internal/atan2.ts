/**
 * v5 Internal: atan2
 * Mechanism: Iterator
 * DispatchKey: atan2
 *
 * 逐元素二参数反正切: atan2(self, other)，返回 [-π, π] 弧度
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.atan2;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function atan2(
    self: ITensorHandle,
    other: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'atan2',
        tensorInputs: [self, other],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
