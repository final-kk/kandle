/**
 * v5 Internal: max.global
 * Mechanism: Iterator
 * DispatchKey: max
 *
 * 全局最大值
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.max_global;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function max_global(
    self: ITensorHandle
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'max',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
