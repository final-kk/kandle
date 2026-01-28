/**
 * v5 Internal: std
 * Mechanism: Iterator
 * DispatchKey: std
 *
 * 沿维度标准差
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.std;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function std(
    self: ITensorHandle,
    dim?: number | readonly number[] | undefined,
    correction?: number,
    keepdim?: boolean
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'std',
        tensorInputs: [self],
        scalarArgs: { correction, keepdim } as Record<string, any>,
        metadata: { dim, correction, keepdim },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
