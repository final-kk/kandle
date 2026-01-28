/**
 * v5 Internal: convTranspose2d
 * Mechanism: Window
 * DispatchKey: conv_transpose2d
 *
 * 2D 转置卷积 (反卷积)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.convTranspose2d;
import { dispatchWindow, type OperatorContext } from '../../dispatch/handlers';

export function convTranspose2d(
    input: ITensorHandle,
    weight: ITensorHandle,
    bias?: ITensorHandle | undefined,
    stride?: number | number[],
    padding?: number | number[],
    outputPadding?: number | number[],
    groups?: number,
    dilation?: number | number[]
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'conv_transpose2d',
        tensorInputs: [input, weight, bias].filter((t): t is ITensorHandle => t !== undefined),
        scalarArgs: { groups } as Record<string, any>,
        metadata: { stride, padding, outputPadding, groups, dilation },
    };
    return dispatchWindow(__entry, ctx) as any;
}
