/**
 * v5 Internal: addmm
 * Mechanism: Matrix
 * DispatchKey: addmm
 *
 * out = beta * self + alpha * (mat1 @ mat2)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.addmm;
import { dispatchMatrix, type OperatorContext } from '../../dispatch/handlers';

export function addmm(
    self: ITensorHandle,
    mat1: ITensorHandle,
    mat2: ITensorHandle,
    beta?: number,
    alpha?: number,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'addmm',
        tensorInputs: [self, mat1, mat2],
        scalarArgs: { beta, alpha } as Record<string, any>,
        metadata: { beta, alpha },
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchMatrix(__entry, ctx) as ITensorHandle;
}
