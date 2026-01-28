/**
 * v5 Internal: argsort
 * Mechanism: Sort
 * DispatchKey: argsort
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.argsort;
import { dispatchSort, type OperatorContext } from '../../dispatch/handlers';

export function argsort(
    self: ITensorHandle,
    dim?: number,
    descending?: boolean,
    stable?: boolean
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'argsort',
        tensorInputs: [self],
        scalarArgs: { descending, stable } as Record<string, any>,
        metadata: { dim, descending, stable },
    };
    return dispatchSort(__entry, ctx) as ITensorHandle;
}
