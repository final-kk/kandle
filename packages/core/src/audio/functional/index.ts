/**
 * Audio Functional API
 *
 * 对标 torchaudio.functional
 *
 * 提供音频处理的函数式 API
 */

// Spectrogram
export { spectrogram, inverseSpectrogram } from './spectrogram';

// Utility functions
export { amplitudeToDB, DBToAmplitude } from './utils';

// Filterbank
export {
    melscaleFbanks,
    linearFbanks,
    createDct,
    type MelScaleType,
    type NormType,
} from './filterbank';

// Delta coefficients
export { computeDeltas } from './deltas';

// Phase processing
export { phaseVocoder } from './phaseVocoder';
export { griffinlim } from './griffinlim';

// Resampling
export {
    resample,
    gcd,
    getSincResampleKernel,
    applySincResampleKernel,
    type ResampleOptions,
} from './resample';

// Pitch shift
export { pitchShift, type PitchShiftOptions } from './pitchShift';

// μ-law encoding/decoding
export { muLawEncoding, muLawDecoding } from './muLaw';

// Emphasis
export { preemphasis, deemphasis } from './emphasis';

// IIR Filtering
export { lfilter, type LfilterOptions } from './lfilter';
export {
    biquad,
    lowpassBiquad,
    highpassBiquad,
    bandpassBiquad,
    bandrejectBiquad,
    allpassBiquad,
    bassBiquad,
    trebleBiquad,
    equalizerBiquad,
} from './biquad';
