/**
 * v5 Internal: istft
 * Mechanism: Composite
 * DispatchKey: istft
 *
 * Computes the inverse Short-Time Fourier Transform to reconstruct the signal.
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.istft;
import { dispatchComposite, type OperatorContext } from '../../dispatch/handlers';

export function istft(
    input: ITensorHandle,
    n_fft: number,
    hop_length?: number | undefined,
    win_length?: number | undefined,
    window?: ITensorHandle | undefined,
    center?: boolean,
    normalized?: boolean,
    onesided?: boolean | undefined,
    length?: number | undefined,
    return_complex?: boolean
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'istft',
        tensorInputs: [input, window].filter((t): t is ITensorHandle => t !== undefined),
        scalarArgs: { n_fft, hop_length, win_length, center, normalized, onesided, length, return_complex } as Record<string, any>,
        metadata: { n_fft, hop_length, win_length, center, normalized, onesided, length, return_complex },
    };
    return dispatchComposite(__entry, ctx) as ITensorHandle;
}
