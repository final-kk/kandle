/**
 * v5 Internal: multinomial
 * Mechanism: Factory
 * DispatchKey: multinomial
 *
 * 从多项式分布中采样索引
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.multinomial;
import { dispatchFactory, type OperatorContext } from '../../dispatch/handlers';

export function multinomial(
    input: ITensorHandle,
    numSamples: number,
    replacement?: boolean
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'multinomial',
        tensorInputs: [input],
        scalarArgs: { numSamples, replacement } as Record<string, any>,
        metadata: { numSamples, replacement },
    };
    return dispatchFactory(__entry, ctx);
}
