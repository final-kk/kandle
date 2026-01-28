/**
 * v5 Internal: ones
 * Mechanism: Factory
 * DispatchKey: ones
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.ones;
import { dispatchFactory, type OperatorContext } from '../../dispatch/handlers';

export function ones(
    size: readonly number[],
    dtype?: string,
    device?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'ones',
        tensorInputs: [],
        scalarArgs: {},
        metadata: { size, dtype, device },
    };
    return dispatchFactory(__entry, ctx);
}
