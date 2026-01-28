/**
 * v5 Internal: randn
 * Mechanism: Factory
 * DispatchKey: randn
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.randn;
import { dispatchFactory, type OperatorContext } from '../../dispatch/handlers';

export function randn(
    size: readonly number[],
    dtype?: string,
    device?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'randn',
        tensorInputs: [],
        scalarArgs: {},
        metadata: { size, dtype, device },
    };
    return dispatchFactory(__entry, ctx);
}
