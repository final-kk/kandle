/**
 * v5 Internal: sigmoid
 * Mechanism: Iterator
 * DispatchKey: sigmoid
 *
 * 逐元素 Sigmoid: 1 / (1 + exp(-self))
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.sigmoid;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function sigmoid(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'sigmoid',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
