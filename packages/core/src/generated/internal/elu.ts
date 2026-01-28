/**
 * v5 Internal: elu
 * Mechanism: Iterator
 * DispatchKey: elu
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.elu;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function elu(
    self: ITensorHandle,
    alpha?: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'elu',
        tensorInputs: [self],
        scalarArgs: { alpha } as Record<string, any>,
        metadata: { alpha },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
