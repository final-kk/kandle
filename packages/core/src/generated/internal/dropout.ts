/**
 * v5 Internal: dropout
 * Mechanism: Iterator
 * DispatchKey: dropout
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.dropout;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function dropout(
    self: ITensorHandle,
    p?: number,
    training?: boolean
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'dropout',
        tensorInputs: [self],
        scalarArgs: { p, training } as Record<string, any>,
        metadata: { p, training },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
