/**
 * Whisper Model Configuration
 *
 * 对标 OpenAI Whisper 和 HuggingFace Transformers WhisperConfig
 *
 * 支持所有 Whisper 尺寸：tiny, base, small, medium, large-v3, large-v3-turbo
 *
 * @module @kandle/model-utils/whisper/config
 */

import type { DType } from '@kandle/types';

// ============================================================================
// Types
// ============================================================================

/**
 * Whisper 模型尺寸
 */
export type WhisperModelSize =
    | 'tiny'
    | 'base'
    | 'small'
    | 'medium'
    | 'large-v3'
    | 'large-v3-turbo';

/**
 * Whisper 模型配置
 *
 * 对应 HuggingFace WhisperConfig
 */
export interface WhisperConfig {
    /** 模型维度 (d_model) */
    dModel: number;

    /** 编码器层数 */
    encoderLayers: number;

    /** 编码器注意力头数 */
    encoderAttentionHeads: number;

    /** 编码器 FFN 维度 */
    encoderFfnDim: number;

    /** 解码器层数 */
    decoderLayers: number;

    /** 解码器注意力头数 */
    decoderAttentionHeads: number;

    /** 解码器 FFN 维度 */
    decoderFfnDim: number;

    /** 词汇表大小 */
    vocabSize: number;

    /** Mel 频率 bins 数量 */
    numMelBins: number;

    /** 编码器最大序列位置 (对应 30 秒音频经过 stride=2 后的帧数) */
    maxSourcePositions: number;

    /** 解码器最大序列位置 */
    maxTargetPositions: number;

    /** 数据类型 */
    dtype?: DType;
}

// ============================================================================
// 预定义配置
// ============================================================================

/**
 * Whisper Tiny 配置
 *
 * 参数: ~39M
 */
export const WHISPER_TINY_CONFIG: WhisperConfig = {
    dModel: 384,
    encoderLayers: 4,
    encoderAttentionHeads: 6,
    encoderFfnDim: 1536,
    decoderLayers: 4,
    decoderAttentionHeads: 6,
    decoderFfnDim: 1536,
    vocabSize: 51865,
    numMelBins: 80,
    maxSourcePositions: 1500,
    maxTargetPositions: 448,
};

/**
 * Whisper Base 配置
 *
 * 参数: ~74M
 */
export const WHISPER_BASE_CONFIG: WhisperConfig = {
    dModel: 512,
    encoderLayers: 6,
    encoderAttentionHeads: 8,
    encoderFfnDim: 2048,
    decoderLayers: 6,
    decoderAttentionHeads: 8,
    decoderFfnDim: 2048,
    vocabSize: 51865,
    numMelBins: 80,
    maxSourcePositions: 1500,
    maxTargetPositions: 448,
};

/**
 * Whisper Small 配置
 *
 * 参数: ~244M
 */
export const WHISPER_SMALL_CONFIG: WhisperConfig = {
    dModel: 768,
    encoderLayers: 12,
    encoderAttentionHeads: 12,
    encoderFfnDim: 3072,
    decoderLayers: 12,
    decoderAttentionHeads: 12,
    decoderFfnDim: 3072,
    vocabSize: 51865,
    numMelBins: 80,
    maxSourcePositions: 1500,
    maxTargetPositions: 448,
};

/**
 * Whisper Medium 配置
 *
 * 参数: ~769M
 */
export const WHISPER_MEDIUM_CONFIG: WhisperConfig = {
    dModel: 1024,
    encoderLayers: 24,
    encoderAttentionHeads: 16,
    encoderFfnDim: 4096,
    decoderLayers: 24,
    decoderAttentionHeads: 16,
    decoderFfnDim: 4096,
    vocabSize: 51865,
    numMelBins: 80,
    maxSourcePositions: 1500,
    maxTargetPositions: 448,
};

/**
 * Whisper Large V3 配置
 *
 * 参数: ~1550M
 * 注意: 使用 128 mel bins（与之前版本不同）
 */
export const WHISPER_LARGE_V3_CONFIG: WhisperConfig = {
    dModel: 1280,
    encoderLayers: 32,
    encoderAttentionHeads: 20,
    encoderFfnDim: 5120,
    decoderLayers: 32,
    decoderAttentionHeads: 20,
    decoderFfnDim: 5120,
    vocabSize: 51866,
    numMelBins: 128,
    maxSourcePositions: 1500,
    maxTargetPositions: 448,
};

/**
 * Whisper Large V3 Turbo 配置
 *
 * 参数: ~809M
 * 注意: Encoder 32 层，Decoder 仅 4 层（蒸馏版本）
 */
export const WHISPER_LARGE_V3_TURBO_CONFIG: WhisperConfig = {
    dModel: 1280,
    encoderLayers: 32,
    encoderAttentionHeads: 20,
    encoderFfnDim: 5120,
    decoderLayers: 4,        // 蒸馏后仅 4 层
    decoderAttentionHeads: 20,
    decoderFfnDim: 5120,
    vocabSize: 51866,
    numMelBins: 128,
    maxSourcePositions: 1500,
    maxTargetPositions: 448,
};

/**
 * 根据模型尺寸获取配置
 */
export function getWhisperConfig(size: WhisperModelSize): WhisperConfig {
    switch (size) {
        case 'tiny':
            return { ...WHISPER_TINY_CONFIG };
        case 'base':
            return { ...WHISPER_BASE_CONFIG };
        case 'small':
            return { ...WHISPER_SMALL_CONFIG };
        case 'medium':
            return { ...WHISPER_MEDIUM_CONFIG };
        case 'large-v3':
            return { ...WHISPER_LARGE_V3_CONFIG };
        case 'large-v3-turbo':
            return { ...WHISPER_LARGE_V3_TURBO_CONFIG };
        default:
            throw new Error(`Unknown Whisper model size: ${size}`);
    }
}

// ============================================================================
// 音频预处理常量
// ============================================================================

/**
 * Whisper 音频预处理参数 (所有模型通用)
 */
export const WHISPER_AUDIO_CONFIG = {
    /** 目标采样率 (Hz) */
    SAMPLE_RATE: 16000,

    /** FFT 大小 (25ms @ 16kHz) */
    N_FFT: 400,

    /** 帧移 (10ms @ 16kHz) */
    HOP_LENGTH: 160,

    /** 音频块长度 (秒) */
    CHUNK_LENGTH: 30,

    /** 30 秒音频的采样点数 */
    N_SAMPLES: 480000,

    /** 30 秒音频的 Mel 帧数 (经过 stride=2 下采样前) */
    N_FRAMES: 3000,
} as const;

// ============================================================================
// 特殊 Token IDs
// ============================================================================

/**
 * Whisper 特殊 Token IDs
 */
export const WHISPER_SPECIAL_TOKENS = {
    /** Start of transcript */
    SOT: 50258,

    /** End of text */
    EOT: 50257,

    /** Start of language tag */
    TRANSLATE: 50358,

    /** Transcribe task */
    TRANSCRIBE: 50359,

    /** Previous text tag */
    PREV: 50361,

    /** No timestamps */
    NO_TIMESTAMPS: 50363,

    /** Timestamp begin (t < 30s) */
    TIMESTAMP_BEGIN: 50364,
} as const;

/**
 * 语言代码到 Token ID 的映射 (部分常用语言)
 */
export const WHISPER_LANGUAGE_TOKENS: Record<string, number> = {
    en: 50259,  // English
    zh: 50260,  // Chinese
    de: 50261,  // German
    es: 50262,  // Spanish
    ru: 50263,  // Russian
    ko: 50264,  // Korean
    fr: 50265,  // French
    ja: 50266,  // Japanese
    pt: 50267,  // Portuguese
    tr: 50268,  // Turkish
    // ... 更多语言可按需添加
};
