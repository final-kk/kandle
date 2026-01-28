/**
 * v5 Internal: conj
 * Mechanism: Iterator
 * DispatchKey: conj
 *
 * 复数共轭: a+bi -> a-bi
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.conj;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function conj(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'conj',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
