/**
 * v5 Internal: blackmanWindow
 * Mechanism: WindowFunc
 * DispatchKey: windowfunc.blackman
 *
 * 生成 Blackman 窗函数
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.blackmanWindow;
import { dispatchWindowFunc, type OperatorContext } from '../../dispatch/handlers';

export function blackmanWindow(
    windowLength: number,
    periodic?: boolean,
    dtype?: string,
    device?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'windowfunc.blackman',
        tensorInputs: [],
        scalarArgs: { windowLength, periodic } as Record<string, any>,
        metadata: { windowLength, periodic, dtype, device },
    };
    return dispatchWindowFunc(__entry, ctx) as ITensorHandle;
}
