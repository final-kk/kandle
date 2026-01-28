/**
 * v5 Internal: fmod.Scalar
 * Mechanism: Iterator
 * DispatchKey: fmod_scalar
 *
 * C++ 风格取模: fmod(self, other) (标量版本)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.fmod_Scalar;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function fmod_Scalar(
    self: ITensorHandle,
    other: number,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'fmod_scalar',
        tensorInputs: [self],
        scalarArgs: { other } as Record<string, any>,
        metadata: { other },
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
