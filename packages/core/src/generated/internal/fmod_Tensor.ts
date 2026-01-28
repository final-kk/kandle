/**
 * v5 Internal: fmod.Tensor
 * Mechanism: Iterator
 * DispatchKey: fmod
 *
 * C++ 风格取模: fmod(self, other)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.fmod_Tensor;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function fmod_Tensor(
    self: ITensorHandle,
    other: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'fmod',
        tensorInputs: [self, other],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
