/**
 * v5 Internal: select
 * Mechanism: View
 * DispatchKey: select
 *
 * 沿指定维度选择单个索引，返回降维的视图
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.select;
import { dispatchView, type OperatorContext } from '../../dispatch/handlers';

export function select(
    self: ITensorHandle,
    dim: number,
    index: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'select',
        tensorInputs: [self],
        scalarArgs: { index } as Record<string, any>,
        metadata: { dim, index },
    };
    return dispatchView(__entry, ctx);
}
