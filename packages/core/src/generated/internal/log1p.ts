/**
 * v5 Internal: log1p
 * Mechanism: Iterator
 * DispatchKey: log1p
 *
 * 逐元素 ln(1 + self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.log1p;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function log1p(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'log1p',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
