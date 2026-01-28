/**
 * v5 Internal: mean
 * Mechanism: Iterator
 * DispatchKey: mean
 *
 * 沿维度求均值
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.mean;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function mean(
    self: ITensorHandle,
    dim?: number | readonly number[] | undefined,
    keepdim?: boolean,
    dtype?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'mean',
        tensorInputs: [self],
        scalarArgs: { keepdim } as Record<string, any>,
        metadata: { dim, keepdim, dtype },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
