/**
 * v5 Internal: abs
 * Mechanism: Iterator
 * DispatchKey: abs
 *
 * 逐元素绝对值: |self|
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.abs;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function abs(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'abs',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
