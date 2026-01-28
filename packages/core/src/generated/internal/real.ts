/**
 * v5 Internal: real
 * Mechanism: Iterator
 * DispatchKey: real
 *
 * 取实部: a+bi -> a
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.real;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function real(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'real',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
