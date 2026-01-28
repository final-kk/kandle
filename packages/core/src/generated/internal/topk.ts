/**
 * v5 Internal: topk
 * Mechanism: Sort
 * DispatchKey: topk
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.topk;
import { dispatchSort, type OperatorContext } from '../../dispatch/handlers';

export function topk(
    self: ITensorHandle,
    k: number,
    dim?: number,
    largest?: boolean,
    sorted?: boolean
): [ITensorHandle, ITensorHandle] {
    const ctx: OperatorContext = {
        opName: 'topk',
        tensorInputs: [self],
        scalarArgs: { k, largest, sorted } as Record<string, any>,
        metadata: { k, dim, largest, sorted },
    };
    return dispatchSort(__entry, ctx) as any;
}
