/**
 * v5 Internal: linear
 * Mechanism: Composite
 * DispatchKey: linear
 *
 * 线性变换: y = input @ weight.T + bias
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.linear;
import { dispatchComposite, type OperatorContext } from '../../dispatch/handlers';

export function linear(
    input: ITensorHandle,
    weight: ITensorHandle,
    bias?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'linear',
        tensorInputs: [input, weight, bias].filter((t): t is ITensorHandle => t !== undefined),
        scalarArgs: {},
        metadata: {},
    };
    return dispatchComposite(__entry, ctx) as ITensorHandle;
}
