/**
 * v5 Internal: remainder.Scalar
 * Mechanism: Iterator
 * DispatchKey: remainder_scalar
 *
 * Python 风格取模: remainder(self, other) (标量版本)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.remainder_Scalar;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function remainder_Scalar(
    self: ITensorHandle,
    other: number,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'remainder_scalar',
        tensorInputs: [self],
        scalarArgs: { other } as Record<string, any>,
        metadata: { other },
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
