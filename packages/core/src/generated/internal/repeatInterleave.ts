/**
 * v5 Internal: repeatInterleave
 * Mechanism: Shape
 * DispatchKey: repeat_interleave
 *
 * 沿维度重复每个元素指定次数
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.repeatInterleave;
import { dispatchShape, type OperatorContext } from '../../dispatch/handlers';

export function repeatInterleave(
    self: ITensorHandle,
    repeats: number | ITensorHandle,
    dim?: number | undefined,
    outputSize?: number | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'repeat_interleave',
        tensorInputs: [self],
        scalarArgs: { outputSize } as Record<string, any>,
        metadata: { repeats, dim, outputSize },
    };
    return dispatchShape(__entry, ctx) as ITensorHandle;
}
