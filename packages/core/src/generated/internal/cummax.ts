/**
 * v5 Internal: cummax
 * Mechanism: Iterator
 * DispatchKey: cummax
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.cummax;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function cummax(
    self: ITensorHandle,
    dim: number
): [ITensorHandle, ITensorHandle] {
    const ctx: OperatorContext = {
        opName: 'cummax',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: { dim },
    };
    return dispatchIterator(__entry, ctx) as [ITensorHandle, ITensorHandle];
}
