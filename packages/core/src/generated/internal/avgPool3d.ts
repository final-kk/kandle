/**
 * v5 Internal: avgPool3d
 * Mechanism: Window
 * DispatchKey: avg_pool3d
 *
 * 3D 平均池化
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.avgPool3d;
import { dispatchWindow, type OperatorContext } from '../../dispatch/handlers';

export function avgPool3d(
    input: ITensorHandle,
    kernelSize: number | number[],
    stride?: number | number[] | undefined,
    padding?: number | number[],
    ceilMode?: boolean,
    countIncludePad?: boolean,
    divisorOverride?: number | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'avg_pool3d',
        tensorInputs: [input],
        scalarArgs: { ceilMode, countIncludePad, divisorOverride } as Record<string, any>,
        metadata: { kernelSize, stride, padding, ceilMode, countIncludePad, divisorOverride },
    };
    return dispatchWindow(__entry, ctx) as any;
}
