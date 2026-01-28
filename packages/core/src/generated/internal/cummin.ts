/**
 * v5 Internal: cummin
 * Mechanism: Iterator
 * DispatchKey: cummin
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.cummin;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function cummin(
    self: ITensorHandle,
    dim: number
): [ITensorHandle, ITensorHandle] {
    const ctx: OperatorContext = {
        opName: 'cummin',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: { dim },
    };
    return dispatchIterator(__entry, ctx) as [ITensorHandle, ITensorHandle];
}
