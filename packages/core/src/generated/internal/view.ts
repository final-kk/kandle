/**
 * v5 Internal: view
 * Mechanism: View
 * DispatchKey: view
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.view;
import { dispatchView, type OperatorContext } from '../../dispatch/handlers';

export function view(
    self: ITensorHandle,
    shape: readonly number[]
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'view',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: { shape },
    };
    return dispatchView(__entry, ctx);
}
