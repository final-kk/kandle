/**
 * v5 Internal: isnan
 * Mechanism: Iterator
 * DispatchKey: isnan
 *
 * 逐元素检查是否为 NaN
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.isnan;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function isnan(
    self: ITensorHandle
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'isnan',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
