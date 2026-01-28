/**
 * v5 Internal: logSoftmax
 * Mechanism: Normalize
 * DispatchKey: log_softmax
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.logSoftmax;
import { dispatchNormalize, type OperatorContext } from '../../dispatch/handlers';

export function logSoftmax(
    self: ITensorHandle,
    dim?: number,
    dtype?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'log_softmax',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: { dim, dtype },
    };
    return dispatchNormalize(__entry, ctx) as ITensorHandle;
}
