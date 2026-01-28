/**
 * v5 Internal: batchNorm
 * Mechanism: Normalize
 * DispatchKey: batch_norm
 *
 * Batch Normalization
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.batchNorm;
import { dispatchNormalize, type OperatorContext } from '../../dispatch/handlers';

export function batchNorm(
    self: ITensorHandle,
    runningMean?: ITensorHandle | undefined,
    runningVar?: ITensorHandle | undefined,
    weight?: ITensorHandle | undefined,
    bias?: ITensorHandle | undefined,
    training?: boolean,
    momentum?: number,
    eps?: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'batch_norm',
        tensorInputs: [self, runningMean, runningVar, weight, bias].filter((t): t is ITensorHandle => t !== undefined),
        scalarArgs: { training, momentum, eps } as Record<string, any>,
        metadata: { training, momentum, eps },
    };
    return dispatchNormalize(__entry, ctx) as ITensorHandle;
}
