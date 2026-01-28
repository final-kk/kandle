/**
 * v5 Internal: full
 * Mechanism: Factory
 * DispatchKey: full
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.full;
import { dispatchFactory, type OperatorContext } from '../../dispatch/handlers';

export function full(
    size: readonly number[],
    fillValue: number,
    dtype?: string | undefined,
    device?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'full',
        tensorInputs: [],
        scalarArgs: { fillValue } as Record<string, any>,
        metadata: { size, fillValue, dtype, device },
    };
    return dispatchFactory(__entry, ctx);
}
