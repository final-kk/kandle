/**
 * v5 Internal: selu
 * Mechanism: Iterator
 * DispatchKey: selu
 *
 * SELU 激活
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.selu;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function selu(
    self: ITensorHandle,
    inplace?: boolean
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'selu',
        tensorInputs: [self],
        scalarArgs: { inplace } as Record<string, any>,
        metadata: { inplace },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
