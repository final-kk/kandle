/**
 * v5 Internal: diagonal
 * Mechanism: View
 * DispatchKey: diagonal
 *
 * 获取对角线视图 (Partial View)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.diagonal;
import { dispatchView, type OperatorContext } from '../../dispatch/handlers';

export function diagonal(
    self: ITensorHandle,
    offset?: number,
    dim1?: number,
    dim2?: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'diagonal',
        tensorInputs: [self],
        scalarArgs: { offset } as Record<string, any>,
        metadata: { offset, dim1, dim2 },
    };
    return dispatchView(__entry, ctx);
}
