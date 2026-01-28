/**
 * v5 Internal: linspace
 * Mechanism: Factory
 * DispatchKey: linspace
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.linspace;
import { dispatchFactory, type OperatorContext } from '../../dispatch/handlers';

export function linspace(
    start: number,
    end: number,
    steps: number,
    dtype?: string | undefined,
    device?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'linspace',
        tensorInputs: [],
        scalarArgs: { start, end, steps } as Record<string, any>,
        metadata: { start, end, steps, dtype, device },
    };
    return dispatchFactory(__entry, ctx);
}
