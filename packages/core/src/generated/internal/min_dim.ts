/**
 * v5 Internal: min.dim
 * Mechanism: Iterator
 * DispatchKey: min_dim
 *
 * 沿维度最小值及索引
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.min_dim;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function min_dim(
    self: ITensorHandle,
    dim: number,
    keepdim?: boolean
): [ITensorHandle, ITensorHandle] {
    const ctx: OperatorContext = {
        opName: 'min_dim',
        tensorInputs: [self],
        scalarArgs: { keepdim } as Record<string, any>,
        metadata: { dim, keepdim },
    };
    return dispatchIterator(__entry, ctx) as [ITensorHandle, ITensorHandle];
}
