/**
 * v5 Internal: hardtanh
 * Mechanism: Iterator
 * DispatchKey: hardtanh
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.hardtanh;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function hardtanh(
    self: ITensorHandle,
    minVal?: number,
    maxVal?: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'hardtanh',
        tensorInputs: [self],
        scalarArgs: { minVal, maxVal } as Record<string, any>,
        metadata: { minVal, maxVal },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
