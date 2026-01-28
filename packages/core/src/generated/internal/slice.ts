/**
 * v5 Internal: slice
 * Mechanism: View
 * DispatchKey: slice
 *
 * 张量切片: 使用 Python 风格切片语法或整数索引返回视图
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.slice;
import { dispatchView, type OperatorContext } from '../../dispatch/handlers';

export function slice(
    self: ITensorHandle,
    slices: string | number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'slice',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: { slices },
    };
    return dispatchView(__entry, ctx);
}
