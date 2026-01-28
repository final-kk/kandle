/**
 * v5 Internal: leakyRelu
 * Mechanism: Iterator
 * DispatchKey: leaky_relu
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.leakyRelu;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function leakyRelu(
    self: ITensorHandle,
    negativeSlope?: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'leaky_relu',
        tensorInputs: [self],
        scalarArgs: { negativeSlope } as Record<string, any>,
        metadata: { negativeSlope },
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
