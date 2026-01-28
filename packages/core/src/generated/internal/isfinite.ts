/**
 * v5 Internal: isfinite
 * Mechanism: Iterator
 * DispatchKey: isfinite
 *
 * 逐元素检查是否为有限数
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.isfinite;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function isfinite(
    self: ITensorHandle
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'isfinite',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
