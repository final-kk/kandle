/**
 * v5 Internal: copy_
 * Mechanism: Copy
 * DispatchKey: copy_
 *
 * 原地拷贝: 将 src 拷贝到 self (in-place)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.copy_;
import { dispatchCopy, type OperatorContext } from '../../dispatch/handlers';

export function copy_(
    self: ITensorHandle,
    src: ITensorHandle
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'copy_',
        tensorInputs: [self, src],
        scalarArgs: {},
        metadata: {},
    };
    return dispatchCopy(__entry, ctx);
}
