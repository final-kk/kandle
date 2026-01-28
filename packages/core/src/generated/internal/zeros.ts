/**
 * v5 Internal: zeros
 * Mechanism: Factory
 * DispatchKey: zeros
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.zeros;
import { dispatchFactory, type OperatorContext } from '../../dispatch/handlers';

export function zeros(
    size: readonly number[],
    dtype?: string,
    device?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'zeros',
        tensorInputs: [],
        scalarArgs: {},
        metadata: { size, dtype, device },
    };
    return dispatchFactory(__entry, ctx);
}
