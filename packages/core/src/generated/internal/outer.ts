/**
 * v5 Internal: outer
 * Mechanism: Matrix
 * DispatchKey: outer
 *
 * 向量外积
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.outer;
import { dispatchMatrix, type OperatorContext } from '../../dispatch/handlers';

export function outer(
    self: ITensorHandle,
    vec2: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'outer',
        tensorInputs: [self, vec2],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchMatrix(__entry, ctx) as ITensorHandle;
}
