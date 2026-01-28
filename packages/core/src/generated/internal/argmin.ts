/**
 * v5 Internal: argmin
 * Mechanism: Iterator
 * DispatchKey: argmin
 *
 * 沿维度最小值索引
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.argmin;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function argmin(
    self: ITensorHandle,
    dim?: number | undefined,
    keepdim?: boolean
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'argmin',
        tensorInputs: [self],
        scalarArgs: { keepdim } as Record<string, any>,
        metadata: { dim, keepdim },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
