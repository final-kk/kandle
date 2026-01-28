/**
 * v5 Internal: triu
 * Mechanism: Triangular
 * DispatchKey: triu
 *
 * 上三角矩阵: 保留 row <= col + diagonal 的元素，其余置零
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.triu;
import { dispatchTriangular, type OperatorContext } from '../../dispatch/handlers';

export function triu(
    self: ITensorHandle,
    diagonal?: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'triu',
        tensorInputs: [self],
        scalarArgs: { diagonal } as Record<string, any>,
        metadata: { diagonal },
    };
    return dispatchTriangular(__entry, ctx) as ITensorHandle;
}
