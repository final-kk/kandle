/**
 * v5 Internal: sign
 * Mechanism: Iterator
 * DispatchKey: sign
 *
 * 逐元素符号: sign(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.sign;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function sign(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'sign',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
