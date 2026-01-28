/**
 * v5 Internal: emptyLike
 * Mechanism: Factory
 * DispatchKey: empty_like
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.emptyLike;
import { dispatchFactory, type OperatorContext } from '../../dispatch/handlers';

export function emptyLike(
    self: ITensorHandle,
    dtype?: string | undefined,
    device?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'empty_like',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: { dtype, device },
    };
    return dispatchFactory(__entry, ctx);
}
