/**
 * v5 Internal: squeeze
 * Mechanism: View
 * DispatchKey: squeeze
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.squeeze;
import { dispatchView, type OperatorContext } from '../../dispatch/handlers';

export function squeeze(
    self: ITensorHandle,
    dim?: number | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'squeeze',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: { dim },
    };
    return dispatchView(__entry, ctx);
}
