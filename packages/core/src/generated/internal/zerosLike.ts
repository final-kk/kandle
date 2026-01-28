/**
 * v5 Internal: zerosLike
 * Mechanism: Factory
 * DispatchKey: zeros_like
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.zerosLike;
import { dispatchFactory, type OperatorContext } from '../../dispatch/handlers';

export function zerosLike(
    self: ITensorHandle,
    dtype?: string | undefined,
    device?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'zeros_like',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: { dtype, device },
    };
    return dispatchFactory(__entry, ctx);
}
