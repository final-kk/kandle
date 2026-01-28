/**
 * v5 Internal: pow.Scalar
 * Mechanism: Iterator
 * DispatchKey: pow_scalar
 *
 * 逐元素幂运算: self ^ exponent (标量版本)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.pow_Scalar;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function pow_Scalar(
    self: ITensorHandle,
    exponent: number,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'pow_scalar',
        tensorInputs: [self],
        scalarArgs: { exponent } as Record<string, any>,
        metadata: { exponent },
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
