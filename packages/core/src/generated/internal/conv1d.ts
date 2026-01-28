/**
 * v5 Internal: conv1d
 * Mechanism: Window
 * DispatchKey: conv1d
 *
 * 1D 卷积操作
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.conv1d;
import { dispatchWindow, type OperatorContext } from '../../dispatch/handlers';

export function conv1d(
    input: ITensorHandle,
    weight: ITensorHandle,
    bias?: ITensorHandle | undefined,
    stride?: number,
    padding?: number | 'same' | 'valid',
    dilation?: number,
    groups?: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'conv1d',
        tensorInputs: [input, weight, bias].filter((t): t is ITensorHandle => t !== undefined),
        scalarArgs: { stride, dilation, groups } as Record<string, any>,
        metadata: { stride, padding, dilation, groups },
    };
    return dispatchWindow(__entry, ctx) as any;
}
