/**
 * v5 Internal: stft
 * Mechanism: Composite
 * DispatchKey: stft
 *
 * Computes the Short-Time Fourier Transform of the input signal.
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.stft;
import { dispatchComposite, type OperatorContext } from '../../dispatch/handlers';

export function stft(
    input: ITensorHandle,
    n_fft: number,
    hop_length?: number | undefined,
    win_length?: number | undefined,
    window?: ITensorHandle | undefined,
    center?: boolean,
    pad_mode?: 'constant' | 'reflect' | 'replicate' | 'circular' | undefined,
    normalized?: boolean,
    onesided?: boolean | undefined,
    return_complex?: boolean
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'stft',
        tensorInputs: [input, window].filter((t): t is ITensorHandle => t !== undefined),
        scalarArgs: { n_fft, hop_length, win_length, center, pad_mode, normalized, onesided, return_complex } as Record<string, any>,
        metadata: { n_fft, hop_length, win_length, center, pad_mode, normalized, onesided, return_complex },
    };
    return dispatchComposite(__entry, ctx) as ITensorHandle;
}
