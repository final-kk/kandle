/**
 * v5 Internal: rfftfreq
 * Mechanism: Composite
 * DispatchKey: rfftfreq
 *
 * Returns the sample frequencies for rfft.
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.rfftfreq;
import { dispatchComposite, type OperatorContext } from '../../dispatch/handlers';

export function rfftfreq(
    n: number,
    d?: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'rfftfreq',
        tensorInputs: [],
        scalarArgs: { n, d } as Record<string, any>,
        metadata: { n, d },
    };
    return dispatchComposite(__entry, ctx) as ITensorHandle;
}
