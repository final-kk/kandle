/**
 * v5 Internal: sum
 * Mechanism: Iterator
 * DispatchKey: sum
 *
 * 沿维度求和
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.sum;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function sum(
    self: ITensorHandle,
    dim?: number | readonly number[] | undefined,
    keepdim?: boolean,
    dtype?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'sum',
        tensorInputs: [self],
        scalarArgs: { keepdim } as Record<string, any>,
        metadata: { dim, keepdim, dtype },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
