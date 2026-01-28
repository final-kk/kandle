/**
 * v5 Internal: maxPool1d
 * Mechanism: Window
 * DispatchKey: max_pool1d
 *
 * 1D 最大池化
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.maxPool1d;
import { dispatchWindow, type OperatorContext } from '../../dispatch/handlers';

export function maxPool1d(
    input: ITensorHandle,
    kernelSize: number,
    stride?: number | undefined,
    padding?: number,
    dilation?: number,
    ceilMode?: boolean,
    returnIndices?: boolean
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'max_pool1d',
        tensorInputs: [input],
        scalarArgs: { kernelSize, stride, padding, dilation, ceilMode, returnIndices } as Record<string, any>,
        metadata: { kernelSize, stride, padding, dilation, ceilMode, returnIndices },
    };
    return dispatchWindow(__entry, ctx) as any;
}
