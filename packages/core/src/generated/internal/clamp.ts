/**
 * v5 Internal: clamp
 * Mechanism: Iterator
 * DispatchKey: clamp
 *
 * 逐元素截断到 [min, max] 范围
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.clamp;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function clamp(
    self: ITensorHandle,
    min?: number | undefined,
    max?: number | undefined,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'clamp',
        tensorInputs: [self],
        scalarArgs: { min, max } as Record<string, any>,
        metadata: { min, max },
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
