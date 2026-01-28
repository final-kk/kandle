/**
 * v5 Internal: layerNorm
 * Mechanism: Normalize
 * DispatchKey: layer_norm
 *
 * Layer Normalization
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.layerNorm;
import { dispatchNormalize, type OperatorContext } from '../../dispatch/handlers';

export function layerNorm(
    self: ITensorHandle,
    normalizedShape: readonly number[],
    weight?: ITensorHandle | undefined,
    bias?: ITensorHandle | undefined,
    eps?: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'layer_norm',
        tensorInputs: [self, weight, bias].filter((t): t is ITensorHandle => t !== undefined),
        scalarArgs: { eps } as Record<string, any>,
        metadata: { normalizedShape, eps },
    };
    return dispatchNormalize(__entry, ctx) as ITensorHandle;
}
