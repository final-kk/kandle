/**
 * v5 Internal: hannWindow
 * Mechanism: WindowFunc
 * DispatchKey: windowfunc.hann
 *
 * 生成 Hann 窗函数
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.hannWindow;
import { dispatchWindowFunc, type OperatorContext } from '../../dispatch/handlers';

export function hannWindow(
    windowLength: number,
    periodic?: boolean,
    dtype?: string,
    device?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'windowfunc.hann',
        tensorInputs: [],
        scalarArgs: { windowLength, periodic } as Record<string, any>,
        metadata: { windowLength, periodic, dtype, device },
    };
    return dispatchWindowFunc(__entry, ctx) as ITensorHandle;
}
