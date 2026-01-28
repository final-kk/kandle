/**
 * v5 Internal: bmm
 * Mechanism: Matrix
 * DispatchKey: bmm
 *
 * 批量矩阵乘法
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.bmm;
import { dispatchMatrix, type OperatorContext } from '../../dispatch/handlers';

export function bmm(
    self: ITensorHandle,
    mat2: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'bmm',
        tensorInputs: [self, mat2],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchMatrix(__entry, ctx) as ITensorHandle;
}
