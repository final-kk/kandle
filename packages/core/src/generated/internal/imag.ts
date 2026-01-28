/**
 * v5 Internal: imag
 * Mechanism: Iterator
 * DispatchKey: imag
 *
 * 取虚部: a+bi -> b
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.imag;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function imag(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'imag',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
