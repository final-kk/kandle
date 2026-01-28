/**
 * v5 Internal: sort
 * Mechanism: Sort
 * DispatchKey: sort
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.sort;
import { dispatchSort, type OperatorContext } from '../../dispatch/handlers';

export function sort(
    self: ITensorHandle,
    dim?: number,
    descending?: boolean,
    stable?: boolean
): [ITensorHandle, ITensorHandle] {
    const ctx: OperatorContext = {
        opName: 'sort',
        tensorInputs: [self],
        scalarArgs: { descending, stable } as Record<string, any>,
        metadata: { dim, descending, stable },
    };
    return dispatchSort(__entry, ctx) as any;
}
