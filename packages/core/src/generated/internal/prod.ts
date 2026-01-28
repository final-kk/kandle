/**
 * v5 Internal: prod
 * Mechanism: Iterator
 * DispatchKey: prod
 *
 * 沿维度求积
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.prod;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function prod(
    self: ITensorHandle,
    dim?: number | undefined,
    keepdim?: boolean,
    dtype?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'prod',
        tensorInputs: [self],
        scalarArgs: { keepdim } as Record<string, any>,
        metadata: { dim, keepdim, dtype },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
