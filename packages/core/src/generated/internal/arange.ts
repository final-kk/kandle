/**
 * v5 Internal: arange
 * Mechanism: Factory
 * DispatchKey: arange
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.arange;
import { dispatchFactory, type OperatorContext } from '../../dispatch/handlers';

export function arange(
    start: number,
    end?: number | undefined,
    step?: number,
    dtype?: string | undefined,
    device?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'arange',
        tensorInputs: [],
        scalarArgs: { start, end, step } as Record<string, any>,
        metadata: { start, end, step, dtype, device },
    };
    return dispatchFactory(__entry, ctx);
}
