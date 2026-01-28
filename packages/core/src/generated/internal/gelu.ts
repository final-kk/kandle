/**
 * v5 Internal: gelu
 * Mechanism: Iterator
 * DispatchKey: gelu
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.gelu;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function gelu(
    self: ITensorHandle,
    approximate?: 'none' | 'tanh'
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'gelu',
        tensorInputs: [self],
        scalarArgs: { approximate } as Record<string, any>,
        metadata: { approximate },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
