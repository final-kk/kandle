/**
 * v5 Internal: kaiserWindow
 * Mechanism: WindowFunc
 * DispatchKey: windowfunc.kaiser
 *
 * 生成 Kaiser 窗函数 (需要 Bessel I₀ 函数)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.kaiserWindow;
import { dispatchWindowFunc, type OperatorContext } from '../../dispatch/handlers';

export function kaiserWindow(
    windowLength: number,
    periodic?: boolean,
    beta?: number,
    dtype?: string,
    device?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'windowfunc.kaiser',
        tensorInputs: [],
        scalarArgs: { windowLength, periodic, beta } as Record<string, any>,
        metadata: { windowLength, periodic, beta, dtype, device },
    };
    return dispatchWindowFunc(__entry, ctx) as ITensorHandle;
}
