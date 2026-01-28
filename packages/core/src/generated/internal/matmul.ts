/**
 * v5 Internal: matmul
 * Mechanism: Matrix
 * DispatchKey: matmul
 *
 * 矩阵乘法 (支持批量)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.matmul;
import { dispatchMatrix, type OperatorContext } from '../../dispatch/handlers';

export function matmul(
    self: ITensorHandle,
    other: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'matmul',
        tensorInputs: [self, other],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchMatrix(__entry, ctx) as ITensorHandle;
}
