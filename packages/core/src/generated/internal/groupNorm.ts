/**
 * v5 Internal: groupNorm
 * Mechanism: Normalize
 * DispatchKey: group_norm
 *
 * Group Normalization
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.groupNorm;
import { dispatchNormalize, type OperatorContext } from '../../dispatch/handlers';

export function groupNorm(
    self: ITensorHandle,
    numGroups: number,
    weight?: ITensorHandle | undefined,
    bias?: ITensorHandle | undefined,
    eps?: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'group_norm',
        tensorInputs: [self, weight, bias].filter((t): t is ITensorHandle => t !== undefined),
        scalarArgs: { numGroups, eps } as Record<string, any>,
        metadata: { numGroups, eps },
    };
    return dispatchNormalize(__entry, ctx) as ITensorHandle;
}
