/**
 * v5 Internal: rmsNorm
 * Mechanism: Normalize
 * DispatchKey: rms_norm
 *
 * RMS Normalization
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.rmsNorm;
import { dispatchNormalize, type OperatorContext } from '../../dispatch/handlers';

export function rmsNorm(
    self: ITensorHandle,
    normalizedShape: readonly number[],
    weight?: ITensorHandle | undefined,
    eps?: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'rms_norm',
        tensorInputs: [self, weight].filter((t): t is ITensorHandle => t !== undefined),
        scalarArgs: { eps } as Record<string, any>,
        metadata: { normalizedShape, eps },
    };
    return dispatchNormalize(__entry, ctx) as ITensorHandle;
}
