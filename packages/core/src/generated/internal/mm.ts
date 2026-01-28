/**
 * v5 Internal: mm
 * Mechanism: Matrix
 * DispatchKey: mm
 *
 * 2D 矩阵乘法
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.mm;
import { dispatchMatrix, type OperatorContext } from '../../dispatch/handlers';

export function mm(
    self: ITensorHandle,
    mat2: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'mm',
        tensorInputs: [self, mat2],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchMatrix(__entry, ctx) as ITensorHandle;
}
