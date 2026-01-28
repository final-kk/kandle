/**
 * v5 Internal: hammingWindow
 * Mechanism: WindowFunc
 * DispatchKey: windowfunc.hamming
 *
 * 生成 Hamming 窗函数
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.hammingWindow;
import { dispatchWindowFunc, type OperatorContext } from '../../dispatch/handlers';

export function hammingWindow(
    windowLength: number,
    periodic?: boolean,
    alpha?: number,
    beta?: number,
    dtype?: string,
    device?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'windowfunc.hamming',
        tensorInputs: [],
        scalarArgs: { windowLength, periodic, alpha, beta } as Record<string, any>,
        metadata: { windowLength, periodic, alpha, beta, dtype, device },
    };
    return dispatchWindowFunc(__entry, ctx) as ITensorHandle;
}
