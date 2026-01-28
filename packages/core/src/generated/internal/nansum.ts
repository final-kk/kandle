/**
 * v5 Internal: nansum
 * Mechanism: Iterator
 * DispatchKey: nansum
 *
 * 沿维度求和(忽略NaN)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.nansum;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function nansum(
    self: ITensorHandle,
    dim?: number | readonly number[] | undefined,
    keepdim?: boolean,
    dtype?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'nansum',
        tensorInputs: [self],
        scalarArgs: { keepdim } as Record<string, any>,
        metadata: { dim, keepdim, dtype },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
