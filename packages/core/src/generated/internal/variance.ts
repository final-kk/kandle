/**
 * v5 Internal: variance
 * Mechanism: Iterator
 * DispatchKey: variance
 *
 * 沿维度方差
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.variance;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function variance(
    self: ITensorHandle,
    dim?: number | readonly number[] | undefined,
    correction?: number,
    keepdim?: boolean
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'variance',
        tensorInputs: [self],
        scalarArgs: { correction, keepdim } as Record<string, any>,
        metadata: { dim, correction, keepdim },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
