/**
 * v5 Internal: max.dim
 * Mechanism: Iterator
 * DispatchKey: max_dim
 *
 * 沿维度最大值及索引
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.max_dim;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function max_dim(
    self: ITensorHandle,
    dim: number,
    keepdim?: boolean
): [ITensorHandle, ITensorHandle] {
    const ctx: OperatorContext = {
        opName: 'max_dim',
        tensorInputs: [self],
        scalarArgs: { keepdim } as Record<string, any>,
        metadata: { dim, keepdim },
    };
    return dispatchIterator(__entry, ctx) as [ITensorHandle, ITensorHandle];
}
