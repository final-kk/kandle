/**
 * v5 Internal: adaptiveMaxPool2d
 * Mechanism: Window
 * DispatchKey: adaptive_max_pool2d
 *
 * 自适应 2D 最大池化
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.adaptiveMaxPool2d;
import { dispatchWindow, type OperatorContext } from '../../dispatch/handlers';

export function adaptiveMaxPool2d(
    input: ITensorHandle,
    outputSize: number | number[],
    returnIndices?: boolean
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'adaptive_max_pool2d',
        tensorInputs: [input],
        scalarArgs: { returnIndices } as Record<string, any>,
        metadata: { outputSize, returnIndices },
    };
    return dispatchWindow(__entry, ctx) as any;
}
