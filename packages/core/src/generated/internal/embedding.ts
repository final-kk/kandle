/**
 * v5 Internal: embedding
 * Mechanism: Composite
 * DispatchKey: embedding
 *
 * 嵌入查找: output[...] = weight[input[...], :]
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.embedding;
import { dispatchComposite, type OperatorContext } from '../../dispatch/handlers';

export function embedding(
    input: ITensorHandle,
    weight: ITensorHandle,
    paddingIdx?: number | undefined,
    maxNorm?: number | undefined,
    normType?: number,
    scaleGradByFreq?: boolean,
    sparse?: boolean
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'embedding',
        tensorInputs: [input, weight],
        scalarArgs: { paddingIdx, maxNorm, normType, scaleGradByFreq, sparse } as Record<string, any>,
        metadata: { paddingIdx, maxNorm, normType, scaleGradByFreq, sparse },
    };
    return dispatchComposite(__entry, ctx) as ITensorHandle;
}
