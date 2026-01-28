/**
 * v5 Internal: isinf
 * Mechanism: Iterator
 * DispatchKey: isinf
 *
 * 逐元素检查是否为无穷大
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.isinf;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function isinf(
    self: ITensorHandle
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'isinf',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
