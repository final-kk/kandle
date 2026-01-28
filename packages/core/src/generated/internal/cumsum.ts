/**
 * v5 Internal: cumsum
 * Mechanism: Iterator
 * DispatchKey: cumsum
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.cumsum;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function cumsum(
    self: ITensorHandle,
    dim: number,
    dtype?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'cumsum',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: { dim, dtype },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
