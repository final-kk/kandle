/**
 * v5 Internal: logsumexp
 * Mechanism: Iterator
 * DispatchKey: logsumexp
 *
 * 数值稳定的 log(sum(exp(x)))，公式: max(x) + log(sum(exp(x - max(x))))
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.logsumexp;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function logsumexp(
    self: ITensorHandle,
    dim: number | readonly number[],
    keepdim?: boolean
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'logsumexp',
        tensorInputs: [self],
        scalarArgs: { keepdim } as Record<string, any>,
        metadata: { dim, keepdim },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
