/**
 * v5 Internal: logsigmoid
 * Mechanism: Iterator
 * DispatchKey: logsigmoid
 *
 * LogSigmoid 激活: log(1 / (1 + exp(-self)))
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.logsigmoid;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function logsigmoid(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'logsigmoid',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
