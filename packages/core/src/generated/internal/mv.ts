/**
 * v5 Internal: mv
 * Mechanism: Matrix
 * DispatchKey: mv
 *
 * 矩阵-向量乘法
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.mv;
import { dispatchMatrix, type OperatorContext } from '../../dispatch/handlers';

export function mv(
    self: ITensorHandle,
    vec: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'mv',
        tensorInputs: [self, vec],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchMatrix(__entry, ctx) as ITensorHandle;
}
