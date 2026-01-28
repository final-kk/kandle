/**
 * v5 Internal: cast
 * Mechanism: Copy
 * DispatchKey: cast
 *
 * 类型转换: 将 tensor 转换为指定的 dtype
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.cast;
import { dispatchCopy, type OperatorContext } from '../../dispatch/handlers';

export function cast(
    self: ITensorHandle,
    dtype: string
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'cast',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: { dtype },
    };
    return dispatchCopy(__entry, ctx);
}
