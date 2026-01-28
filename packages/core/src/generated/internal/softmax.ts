/**
 * v5 Internal: softmax
 * Mechanism: Normalize
 * DispatchKey: softmax
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.softmax;
import { dispatchNormalize, type OperatorContext } from '../../dispatch/handlers';

export function softmax(
    self: ITensorHandle,
    dim?: number,
    dtype?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'softmax',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: { dim, dtype },
    };
    return dispatchNormalize(__entry, ctx) as ITensorHandle;
}
