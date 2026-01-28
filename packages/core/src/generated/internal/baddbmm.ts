/**
 * v5 Internal: baddbmm
 * Mechanism: Matrix
 * DispatchKey: baddbmm
 *
 * out = beta * self + alpha * (batch1 @ batch2)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.baddbmm;
import { dispatchMatrix, type OperatorContext } from '../../dispatch/handlers';

export function baddbmm(
    self: ITensorHandle,
    batch1: ITensorHandle,
    batch2: ITensorHandle,
    beta?: number,
    alpha?: number,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'baddbmm',
        tensorInputs: [self, batch1, batch2],
        scalarArgs: { beta, alpha } as Record<string, any>,
        metadata: { beta, alpha },
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchMatrix(__entry, ctx) as ITensorHandle;
}
