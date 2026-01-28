/**
 * v5 Internal: min.global
 * Mechanism: Iterator
 * DispatchKey: min
 *
 * 全局最小值
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.min_global;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function min_global(
    self: ITensorHandle
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'min',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
