/**
 * v5 Internal: any
 * Mechanism: Iterator
 * DispatchKey: any
 *
 * 沿维度逻辑或
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.any;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function any(
    self: ITensorHandle,
    dim?: number | undefined,
    keepdim?: boolean
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'any',
        tensorInputs: [self],
        scalarArgs: { keepdim } as Record<string, any>,
        metadata: { dim, keepdim },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
