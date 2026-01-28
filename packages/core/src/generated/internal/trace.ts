/**
 * v5 Internal: trace
 * Mechanism: Composite
 * DispatchKey: trace
 *
 * 矩阵迹: 对角线元素之和
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.trace;
import { dispatchComposite, type OperatorContext } from '../../dispatch/handlers';

export function trace(
    self: ITensorHandle
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'trace',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
    };
    return dispatchComposite(__entry, ctx) as ITensorHandle;
}
