/**
 * v5 Internal: silu
 * Mechanism: Iterator
 * DispatchKey: silu
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.silu;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function silu(
    self: ITensorHandle
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'silu',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
