/**
 * v5 Internal: norm
 * Mechanism: Iterator
 * DispatchKey: norm
 *
 * 沿维度范数
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.norm;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function norm(
    self: ITensorHandle,
    p?: number,
    dim?: number | readonly number[] | undefined,
    keepdim?: boolean
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'norm',
        tensorInputs: [self],
        scalarArgs: { p, keepdim } as Record<string, any>,
        metadata: { p, dim, keepdim },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
