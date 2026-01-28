/**
 * v5 Internal: remainder.Tensor
 * Mechanism: Iterator
 * DispatchKey: remainder
 *
 * Python 风格取模: remainder(self, other)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.remainder_Tensor;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function remainder_Tensor(
    self: ITensorHandle,
    other: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'remainder',
        tensorInputs: [self, other],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
