/**
 * v5 Internal: maxPool3d
 * Mechanism: Window
 * DispatchKey: max_pool3d
 *
 * 3D 最大池化
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.maxPool3d;
import { dispatchWindow, type OperatorContext } from '../../dispatch/handlers';

export function maxPool3d(
    input: ITensorHandle,
    kernelSize: number | number[],
    stride?: number | number[] | undefined,
    padding?: number | number[],
    dilation?: number | number[],
    ceilMode?: boolean,
    returnIndices?: boolean
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'max_pool3d',
        tensorInputs: [input],
        scalarArgs: { ceilMode, returnIndices } as Record<string, any>,
        metadata: { kernelSize, stride, padding, dilation, ceilMode, returnIndices },
    };
    return dispatchWindow(__entry, ctx) as any;
}
