/**
 * v5 Internal: onesLike
 * Mechanism: Factory
 * DispatchKey: ones_like
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.onesLike;
import { dispatchFactory, type OperatorContext } from '../../dispatch/handlers';

export function onesLike(
    self: ITensorHandle,
    dtype?: string | undefined,
    device?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'ones_like',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: { dtype, device },
    };
    return dispatchFactory(__entry, ctx);
}
