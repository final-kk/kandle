/**
 * v5 Internal: contiguous
 * Mechanism: Copy
 * DispatchKey: contiguous
 *
 * 确保 tensor 按指定格式连续存储。如果已是目标格式则返回原 tensor。
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.contiguous;
import { dispatchCopy, type OperatorContext } from '../../dispatch/handlers';

export function contiguous(
    self: ITensorHandle,
    memoryFormat?: unknown | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'contiguous',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: { memoryFormat },
    };
    return dispatchCopy(__entry, ctx);
}
