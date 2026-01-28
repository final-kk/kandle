/**
 * v5 Internal: empty
 * Mechanism: Factory
 * DispatchKey: empty
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.empty;
import { dispatchFactory, type OperatorContext } from '../../dispatch/handlers';

export function empty(
    size: readonly number[],
    dtype?: string,
    device?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'empty',
        tensorInputs: [],
        scalarArgs: {},
        metadata: { size, dtype, device },
    };
    return dispatchFactory(__entry, ctx);
}
