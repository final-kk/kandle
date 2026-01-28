/**
 * v5 Internal: rand
 * Mechanism: Factory
 * DispatchKey: rand
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.rand;
import { dispatchFactory, type OperatorContext } from '../../dispatch/handlers';

export function rand(
    size: readonly number[],
    dtype?: string,
    device?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'rand',
        tensorInputs: [],
        scalarArgs: {},
        metadata: { size, dtype, device },
    };
    return dispatchFactory(__entry, ctx);
}
