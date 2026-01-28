/**
 * v5 Internal: to
 * Mechanism: Copy
 * DispatchKey: to
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.to;
import { dispatchCopy, type OperatorContext } from '../../dispatch/handlers';

export function to(
    self: ITensorHandle,
    dtype?: string | undefined,
    device?: string | undefined,
    copy?: boolean
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'to',
        tensorInputs: [self],
        scalarArgs: { copy } as Record<string, any>,
        metadata: { dtype, device, copy },
    };
    return dispatchCopy(__entry, ctx);
}
