/**
 * v5 Internal: asStrided
 * Mechanism: View
 * DispatchKey: as_strided
 *
 * 创建具有指定 size 和 stride 的视图（STFT 分帧等场景的核心操作）
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.asStrided;
import { dispatchView, type OperatorContext } from '../../dispatch/handlers';

export function asStrided(
    self: ITensorHandle,
    size: readonly number[],
    stride: readonly number[],
    storageOffset?: number | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'as_strided',
        tensorInputs: [self],
        scalarArgs: { storageOffset } as Record<string, any>,
        metadata: { size, stride, storageOffset },
    };
    return dispatchView(__entry, ctx);
}
