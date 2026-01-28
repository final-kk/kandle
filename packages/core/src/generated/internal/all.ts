/**
 * v5 Internal: all
 * Mechanism: Iterator
 * DispatchKey: all
 *
 * 沿维度逻辑与
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.all;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function all(
    self: ITensorHandle,
    dim?: number | undefined,
    keepdim?: boolean
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'all',
        tensorInputs: [self],
        scalarArgs: { keepdim } as Record<string, any>,
        metadata: { dim, keepdim },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
