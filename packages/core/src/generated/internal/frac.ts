/**
 * v5 Internal: frac
 * Mechanism: Iterator
 * DispatchKey: frac
 *
 * 逐元素取小数部分: frac(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.frac;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function frac(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'frac',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
