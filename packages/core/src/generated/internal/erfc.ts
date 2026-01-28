/**
 * v5 Internal: erfc
 * Mechanism: Iterator
 * DispatchKey: erfc
 *
 * 逐元素互补误差函数: erfc(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.erfc;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function erfc(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'erfc',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
