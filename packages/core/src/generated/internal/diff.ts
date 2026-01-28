/**
 * v5 Internal: diff
 * Mechanism: Shape
 * DispatchKey: diff
 *
 * 计算 N 阶前向差分: out[i] = input[i+1] - input[i]
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.diff;
import { dispatchShape, type OperatorContext } from '../../dispatch/handlers';

export function diff(
    self: ITensorHandle,
    n?: number,
    dim?: number,
    prepend?: ITensorHandle | undefined,
    append?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'diff',
        tensorInputs: [self, prepend, append].filter((t): t is ITensorHandle => t !== undefined),
        scalarArgs: { n } as Record<string, any>,
        metadata: { n, dim },
    };
    return dispatchShape(__entry, ctx) as ITensorHandle;
}
