/**
 * v5 Internal: flip
 * Mechanism: Shape
 * DispatchKey: flip
 *
 * 沿给定维度翻转张量元素顺序
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.flip;
import { dispatchShape, type OperatorContext } from '../../dispatch/handlers';

export function flip(
    self: ITensorHandle,
    dims: number | readonly number[]
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'flip',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: { dims },
    };
    return dispatchShape(__entry, ctx) as ITensorHandle;
}
