/**
 * v5 Internal: nanmean
 * Mechanism: Iterator
 * DispatchKey: nanmean
 *
 * 沿维度求均值(忽略NaN)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.nanmean;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function nanmean(
    self: ITensorHandle,
    dim?: number | readonly number[] | undefined,
    keepdim?: boolean,
    dtype?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'nanmean',
        tensorInputs: [self],
        scalarArgs: { keepdim } as Record<string, any>,
        metadata: { dim, keepdim, dtype },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
