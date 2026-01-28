/**
 * Audio Transforms API
 *
 * 对标 torchaudio.transforms
 *
 * 提供音频处理的类式 API
 */

// Spectrogram
export { Spectrogram, InverseSpectrogram, type SpectrogramOptions, type InverseSpectrogramOptions } from './Spectrogram';

// MelScale
export { MelScale, InverseMelScale, type MelScaleOptions, type InverseMelScaleOptions } from './MelScale';

// MelSpectrogram
export { MelSpectrogram, type MelSpectrogramOptions } from './MelSpectrogram';

// MFCC
export { MFCC, type MFCCOptions } from './MFCC';

// LFCC
export { LFCC, type LFCCOptions } from './LFCC';

// AmplitudeToDB
export { AmplitudeToDB, type AmplitudeToDBOptions } from './AmplitudeToDB';

// SpecAugment
export { FrequencyMasking, TimeMasking, type FrequencyMaskingOptions, type TimeMaskingOptions } from './augment';

// Phase processing
export { GriffinLim, type GriffinLimOptions } from './GriffinLim';
export { TimeStretch, type TimeStretchOptions } from './TimeStretch';

// Resampling
export { Resample, type ResampleTransformOptions } from './Resample';

// Pitch shift
export { PitchShift, type PitchShiftTransformOptions } from './PitchShift';

// μ-law encoding/decoding
export { MuLawEncoding, MuLawDecoding, type MuLawEncodingOptions } from './MuLawEncoding';

// Re-export types from functional for convenience
export type { MelScaleType, NormType } from '../functional';
