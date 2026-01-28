/**
 * v5 Internal: cumprod
 * Mechanism: Iterator
 * DispatchKey: cumprod
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.cumprod;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function cumprod(
    self: ITensorHandle,
    dim: number,
    dtype?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'cumprod',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: { dim, dtype },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
