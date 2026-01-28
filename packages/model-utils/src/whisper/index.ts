/**
 * Whisper Model Components - Whisper 语音识别模型组件
 *
 * @module @kandle/model-utils/whisper
 */

// Config
export {
    type WhisperModelSize,
    type WhisperConfig,
    getWhisperConfig,
    WHISPER_TINY_CONFIG,
    WHISPER_BASE_CONFIG,
    WHISPER_SMALL_CONFIG,
    WHISPER_MEDIUM_CONFIG,
    WHISPER_LARGE_V3_CONFIG,
    WHISPER_LARGE_V3_TURBO_CONFIG,
    WHISPER_AUDIO_CONFIG,
    WHISPER_SPECIAL_TOKENS,
    WHISPER_LANGUAGE_TOKENS,
} from './config';

// Building Blocks
export {
    ResidualAttentionBlock,
    type ResidualAttentionBlockConfig,
    type ResidualAttentionBlockForwardOptions,
} from './block';

// Encoder
export {
    WhisperAudioEncoder,
    type WhisperAudioEncoderConfig,
} from './encoder';

// Decoder
export {
    WhisperTextDecoder,
    type WhisperTextDecoderConfig,
    type WhisperTextDecoderForwardOptions,
} from './decoder';

// Full Model
export {
    WhisperModel,
    type WhisperModelForwardOptions,
    type WhisperGenerateOptions,
} from './model';

// HuggingFace Weight Loader
export {
    processHFWhisperWeights,
    applyWeightsToModel,
    type ProcessedWeights,
    type HFWeightLoadResult,
} from './hf-loader';
