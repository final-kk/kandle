/**
 * v5 Internal: addmv
 * Mechanism: Matrix
 * DispatchKey: addmv
 *
 * out = beta * self + alpha * (mat @ vec)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.addmv;
import { dispatchMatrix, type OperatorContext } from '../../dispatch/handlers';

export function addmv(
    self: ITensorHandle,
    mat: ITensorHandle,
    vec: ITensorHandle,
    beta?: number,
    alpha?: number,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'addmv',
        tensorInputs: [self, mat, vec],
        scalarArgs: { beta, alpha } as Record<string, any>,
        metadata: { beta, alpha },
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchMatrix(__entry, ctx) as ITensorHandle;
}
