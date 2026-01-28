/**
 * v5 Internal: softmin
 * Mechanism: Normalize
 * DispatchKey: softmin
 *
 * Softmin 激活: softmax(-self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.softmin;
import { dispatchNormalize, type OperatorContext } from '../../dispatch/handlers';

export function softmin(
    self: ITensorHandle,
    dim?: number,
    dtype?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'softmin',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: { dim, dtype },
    };
    return dispatchNormalize(__entry, ctx) as ITensorHandle;
}
