/**
 * v5 Internal: normalize
 * Mechanism: Normalize
 * DispatchKey: lp_normalize
 *
 * F.normalize: self / self.norm()
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.normalize;
import { dispatchNormalize, type OperatorContext } from '../../dispatch/handlers';

export function normalize(
    self: ITensorHandle,
    p?: number,
    dim?: number,
    eps?: number,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'lp_normalize',
        tensorInputs: [self],
        scalarArgs: { p, eps } as Record<string, any>,
        metadata: { p, dim, eps },
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchNormalize(__entry, ctx) as ITensorHandle;
}
