/**
 * v5 Internal: conv3d
 * Mechanism: Window
 * DispatchKey: conv3d
 *
 * 3D 卷积操作
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.conv3d;
import { dispatchWindow, type OperatorContext } from '../../dispatch/handlers';

export function conv3d(
    input: ITensorHandle,
    weight: ITensorHandle,
    bias?: ITensorHandle | undefined,
    stride?: number | number[],
    padding?: number | number[] | 'same' | 'valid',
    dilation?: number | number[],
    groups?: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'conv3d',
        tensorInputs: [input, weight, bias].filter((t): t is ITensorHandle => t !== undefined),
        scalarArgs: { groups } as Record<string, any>,
        metadata: { stride, padding, dilation, groups },
    };
    return dispatchWindow(__entry, ctx) as any;
}
