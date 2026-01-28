/**
 * v5 Internal: expm1
 * Mechanism: Iterator
 * DispatchKey: expm1
 *
 * 逐元素 exp(self) - 1
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.expm1;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function expm1(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'expm1',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
