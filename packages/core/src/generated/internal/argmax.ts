/**
 * v5 Internal: argmax
 * Mechanism: Iterator
 * DispatchKey: argmax
 *
 * 沿维度最大值索引
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.argmax;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function argmax(
    self: ITensorHandle,
    dim?: number | undefined,
    keepdim?: boolean
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'argmax',
        tensorInputs: [self],
        scalarArgs: { keepdim } as Record<string, any>,
        metadata: { dim, keepdim },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
