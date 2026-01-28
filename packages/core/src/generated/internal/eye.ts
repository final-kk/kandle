/**
 * v5 Internal: eye
 * Mechanism: Factory
 * DispatchKey: eye
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.eye;
import { dispatchFactory, type OperatorContext } from '../../dispatch/handlers';

export function eye(
    n: number,
    m?: number | undefined,
    dtype?: string,
    device?: string | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'eye',
        tensorInputs: [],
        scalarArgs: { n, m } as Record<string, any>,
        metadata: { n, m, dtype, device },
    };
    return dispatchFactory(__entry, ctx);
}
