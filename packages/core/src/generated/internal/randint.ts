/**
 * v5 Internal: randint
 * Mechanism: Factory
 * DispatchKey: randint
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.randint;
import { dispatchFactory, type OperatorContext } from '../../dispatch/handlers';

export function randint(
    low: number,
    high: number,
    size: readonly number[],
    dtype?: string,
    device?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'randint',
        tensorInputs: [],
        scalarArgs: { low, high } as Record<string, any>,
        metadata: { low, high, size, dtype, device },
    };
    return dispatchFactory(__entry, ctx);
}
