/**
 * v5 Internal: dot
 * Mechanism: Matrix
 * DispatchKey: dot
 *
 * 向量点积
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.dot;
import { dispatchMatrix, type OperatorContext } from '../../dispatch/handlers';

export function dot(
    self: ITensorHandle,
    other: ITensorHandle
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'dot',
        tensorInputs: [self, other],
        scalarArgs: {},
        metadata: {},
    };
    return dispatchMatrix(__entry, ctx) as ITensorHandle;
}
