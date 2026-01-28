/**
 * v5 Internal: maxPool2d
 * Mechanism: Window
 * DispatchKey: max_pool2d
 *
 * 2D 最大池化
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.maxPool2d;
import { dispatchWindow, type OperatorContext } from '../../dispatch/handlers';

export function maxPool2d(
    input: ITensorHandle,
    kernelSize: number | number[],
    stride?: number | number[] | undefined,
    padding?: number | number[],
    dilation?: number | number[],
    ceilMode?: boolean,
    returnIndices?: boolean
): ITensorHandle | [ITensorHandle, ITensorHandle] {
    const ctx: OperatorContext = {
        opName: 'max_pool2d',
        tensorInputs: [input],
        scalarArgs: { ceilMode, returnIndices } as Record<string, any>,
        metadata: { kernelSize, stride, padding, dilation, ceilMode, returnIndices },
    };
    return dispatchWindow(__entry, ctx) as any;
}
