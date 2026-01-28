/**
 * v5 Internal: bartlettWindow
 * Mechanism: WindowFunc
 * DispatchKey: windowfunc.bartlett
 *
 * 生成 Bartlett 窗函数
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.bartlettWindow;
import { dispatchWindowFunc, type OperatorContext } from '../../dispatch/handlers';

export function bartlettWindow(
    windowLength: number,
    periodic?: boolean,
    dtype?: string,
    device?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'windowfunc.bartlett',
        tensorInputs: [],
        scalarArgs: { windowLength, periodic } as Record<string, any>,
        metadata: { windowLength, periodic, dtype, device },
    };
    return dispatchWindowFunc(__entry, ctx) as ITensorHandle;
}
