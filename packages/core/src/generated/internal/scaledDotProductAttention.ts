/**
 * v5 Internal: scaledDotProductAttention
 * Mechanism: Composite
 * DispatchKey: scaled_dot_product_attention
 *
 * Scaled Dot-Product Attention: softmax(Q @ K^T / scale + mask) @ V
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.scaledDotProductAttention;
import { dispatchComposite, type OperatorContext } from '../../dispatch/handlers';

export function scaledDotProductAttention(
    query: ITensorHandle,
    key: ITensorHandle,
    value: ITensorHandle,
    attnMask?: ITensorHandle | undefined,
    dropoutP?: number,
    isCausal?: boolean,
    scale?: number | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'scaled_dot_product_attention',
        tensorInputs: [query, key, value, attnMask].filter((t): t is ITensorHandle => t !== undefined),
        scalarArgs: { dropoutP, isCausal, scale } as Record<string, any>,
        metadata: { dropoutP, isCausal, scale },
    };
    return dispatchComposite(__entry, ctx) as ITensorHandle;
}
