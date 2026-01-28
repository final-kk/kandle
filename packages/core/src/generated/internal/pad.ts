/**
 * v5 Internal: pad
 * Mechanism: Factory
 * DispatchKey: pad
 *
 * N 维张量填充 (STFT center padding 等场景)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.pad;
import { dispatchFactory, type OperatorContext } from '../../dispatch/handlers';

export function pad(
    input: ITensorHandle,
    pad: readonly number[],
    mode?: 'constant' | 'reflect' | 'replicate' | 'circular' | undefined,
    value?: number | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'pad',
        tensorInputs: [input],
        scalarArgs: { mode, value } as Record<string, any>,
        metadata: { pad, mode, value },
    };
    return dispatchFactory(__entry, ctx);
}
