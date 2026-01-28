/**
 * v5 Internal: diag
 * Mechanism: Composite
 * DispatchKey: diag
 *
 * 对角线构造 (1D->2D) 或提取 (2D->1D)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.diag;
import { dispatchComposite, type OperatorContext } from '../../dispatch/handlers';

export function diag(
    self: ITensorHandle,
    diagonal?: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'diag',
        tensorInputs: [self],
        scalarArgs: { diagonal } as Record<string, any>,
        metadata: { diagonal },
    };
    return dispatchComposite(__entry, ctx) as ITensorHandle;
}
