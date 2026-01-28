/**
 * @kandle/model-utils - 模型构建工具包
 *
 * 临时内部验证包，用于验证 kandle 基础设施的可用性。
 * 未来将移至独立仓库，演化为:
 * - @kandle/transformers
 * - @kandle/cnn
 * - @kandle/yolo
 * 等上层库
 *
 * 注意：此包只依赖 @kandle/core，不直接依赖底层包
 *
 * @module @kandle/model-utils
 */

// RoPE (Rotary Position Embedding) - Transformer 位置编码
export {
    RotaryEmbedding,
    type RopeConfig,
    type RopeOutput,
    rotateHalf,
    applyRotaryPosEmb,
    applyRotaryPosEmbDirect,
} from "./rope";

// Sinusoidal Positional Encoding - 原始 Transformer 位置编码
export {
    SinusoidalPositionalEncoding,
    type SinusoidalConfig,
    createSinusoidalEncoding,
} from "./sinusoidal";

// Attention - 注意力机制模块
export {
    GroupedQueryAttention,
    type GroupedQueryAttentionOptions,
    type GroupedQueryAttentionForwardOptions,
} from "./attention";

// MLP - 前馈网络模块
export { SwiGLUMLP, type SwiGLUMLPOptions } from "./mlp";

// Qwen3 Model Components - Qwen3 模型组件
export {
    Qwen3DecoderLayer,
    type Qwen3DecoderLayerConfig,
    type Qwen3DecoderLayerForwardOptions,
    Qwen3Model,
    type Qwen3ModelConfig,
    type Qwen3ModelForwardOptions,
    Qwen3ForCausalLM,
    type GenerationStep,
    type GeneratorConfig,
    DEFAULT_GENERATOR_CONFIG,
    // GenerationSession - 支持前进/后退的生成会话
    GenerationSession,
    type GenerationSessionConfig,
    type SessionGenerationStep,
    // Logit Lens
    type LogitLensConfig,
    type LayerPrediction,
} from "./qwen3";

// KV Cache - 用于 Transformer 推理的缓存
export { StaticKVCache, type StaticKVCacheOptions, type LayerKVCache } from "./kv-cache";

// Sampling - 文本生成采样策略
export {
    sample,
    sampleGreedy,
    type SamplerConfig,
    type SampleResult,
    DEFAULT_SAMPLER_CONFIG,
} from "./sampling";

// Whisper Model Components - Whisper 语音识别模型组件
export {
    // Config
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
    // Building Blocks
    ResidualAttentionBlock,
    type ResidualAttentionBlockConfig,
    type ResidualAttentionBlockForwardOptions,
    // Encoder
    WhisperAudioEncoder,
    type WhisperAudioEncoderConfig,
    // Decoder
    WhisperTextDecoder,
    type WhisperTextDecoderConfig,
    type WhisperTextDecoderForwardOptions,
    // Model
    WhisperModel,
    type WhisperModelForwardOptions,
    type WhisperGenerateOptions,
    // HF Weight Loader
    processHFWhisperWeights,
    applyWeightsToModel,
    type ProcessedWeights,
    type HFWeightLoadResult,
} from "./whisper";
