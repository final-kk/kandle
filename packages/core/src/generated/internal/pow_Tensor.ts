/**
 * v5 Internal: pow.Tensor
 * Mechanism: Iterator
 * DispatchKey: pow
 *
 * 逐元素幂运算: self ^ exponent
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.pow_Tensor;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function pow_Tensor(
    self: ITensorHandle,
    exponent: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'pow',
        tensorInputs: [self, exponent],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
