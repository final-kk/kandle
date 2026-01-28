/**
 * v5 Internal: adaptiveAvgPool2d
 * Mechanism: Window
 * DispatchKey: adaptive_avg_pool2d
 *
 * 自适应 2D 平均池化（自动计算 kernel/stride）
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.adaptiveAvgPool2d;
import { dispatchWindow, type OperatorContext } from '../../dispatch/handlers';

export function adaptiveAvgPool2d(
    input: ITensorHandle,
    outputSize: number | number[]
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'adaptive_avg_pool2d',
        tensorInputs: [input],
        scalarArgs: {},
        metadata: { outputSize },
    };
    return dispatchWindow(__entry, ctx) as any;
}
