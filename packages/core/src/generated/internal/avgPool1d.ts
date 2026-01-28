/**
 * v5 Internal: avgPool1d
 * Mechanism: Window
 * DispatchKey: avg_pool1d
 *
 * 1D 平均池化
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.avgPool1d;
import { dispatchWindow, type OperatorContext } from '../../dispatch/handlers';

export function avgPool1d(
    input: ITensorHandle,
    kernelSize: number,
    stride?: number | undefined,
    padding?: number,
    ceilMode?: boolean,
    countIncludePad?: boolean
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'avg_pool1d',
        tensorInputs: [input],
        scalarArgs: { kernelSize, stride, padding, ceilMode, countIncludePad } as Record<string, any>,
        metadata: { kernelSize, stride, padding, ceilMode, countIncludePad },
    };
    return dispatchWindow(__entry, ctx) as any;
}
