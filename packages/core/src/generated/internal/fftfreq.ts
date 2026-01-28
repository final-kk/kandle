/**
 * v5 Internal: fftfreq
 * Mechanism: Composite
 * DispatchKey: fftfreq
 *
 * Returns the DFT sample frequencies.
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.fftfreq;
import { dispatchComposite, type OperatorContext } from '../../dispatch/handlers';

export function fftfreq(
    n: number,
    d?: number
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'fftfreq',
        tensorInputs: [],
        scalarArgs: { n, d } as Record<string, any>,
        metadata: { n, d },
    };
    return dispatchComposite(__entry, ctx) as ITensorHandle;
}
