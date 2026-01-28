/**
 * v5 Internal: round
 * Mechanism: Iterator
 * DispatchKey: round
 *
 * 逐元素四舍五入: round(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.round;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function round(
    self: ITensorHandle,
    decimals?: number,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'round',
        tensorInputs: [self],
        scalarArgs: { decimals } as Record<string, any>,
        metadata: { decimals },
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
