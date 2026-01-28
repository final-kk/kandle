/**
 * Audio Module
 *
 * 对标 torchaudio
 *
 * 提供音频处理功能，包括：
 * - functional: 函数式 API
 * - transforms: 类式 API
 */

// Functional API
import * as functional from './functional';
export { functional };

// Transforms API
import * as transforms from './transforms';
export { transforms };

// 便捷导出常用类 (模仿 torchaudio 的顶级导出)
export {
    Spectrogram,
    InverseSpectrogram,
    MelSpectrogram,
    MelScale,
    InverseMelScale,
    MFCC,
    LFCC,
    AmplitudeToDB,
    FrequencyMasking,
    TimeMasking,
    GriffinLim,
    TimeStretch,
    // 第三阶段新增
    Resample,
    PitchShift,
    MuLawEncoding,
    MuLawDecoding,
} from './transforms';

