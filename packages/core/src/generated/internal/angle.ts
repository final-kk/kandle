/**
 * v5 Internal: angle
 * Mechanism: Iterator
 * DispatchKey: angle
 *
 * 逐元素计算复数相位角 (弧度)，实数返回 0 或 π
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.angle;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function angle(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'angle',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
