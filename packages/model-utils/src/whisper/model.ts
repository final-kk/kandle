/**
 * WhisperModel - 完整的 Whisper 模型
 *
 * 对标 OpenAI Whisper 模型
 *
 * 结构：
 * - AudioEncoder: 将 Mel 频谱图编码为音频特征
 * - TextDecoder: 自回归生成文本 token
 *
 * @module @kandle/model-utils/whisper/model
 */

import type { DType } from '@kandle/types';
import {
    Tensor,
    nn,
    select,
} from '@kandle/core';

import { WhisperAudioEncoder } from './encoder';
import { WhisperTextDecoder } from './decoder';
import {
    type WhisperConfig,
    type WhisperModelSize,
    getWhisperConfig,
    WHISPER_SPECIAL_TOKENS,
    WHISPER_LANGUAGE_TOKENS,
} from './config';

// ============================================================================
// Types
// ============================================================================

export interface WhisperModelForwardOptions {
    /** 解码器偏移量 (用于增量解码) */
    decoderOffset?: number;
}

export interface WhisperGenerateOptions {
    /** 语言代码 (如 'en', 'zh', 'ja') */
    language?: string;

    /** 任务类型 ('transcribe' 或 'translate') */
    task?: 'transcribe' | 'translate';

    /** 最大生成 token 数 */
    maxNewTokens?: number;

    /** 是否带时间戳 */
    withTimestamps?: boolean;
}

// ============================================================================
// WhisperModel Class
// ============================================================================

/**
 * Whisper 语音识别模型
 *
 * HuggingFace 权重结构：
 * - encoder.conv1.*, encoder.conv2.*
 * - encoder.embed_positions.weight (固定正弦位置编码)
 * - encoder.layers.0.*, ...
 * - encoder.layer_norm.*
 * - decoder.embed_tokens.*, decoder.embed_positions.* (可学习位置编码)
 * - decoder.layers.0.*, ...
 * - decoder.layer_norm.*
 * - (proj_out 与 decoder.embed_tokens 共享权重)
 */
export class WhisperModel extends nn.Module {
    // Config
    readonly config: WhisperConfig;

    // Encoder & Decoder
    encoder: WhisperAudioEncoder;
    decoder: WhisperTextDecoder;

    constructor(config: WhisperConfig) {
        super();

        this.config = config;

        // 创建 Encoder
        this.encoder = WhisperAudioEncoder.fromConfig(config);

        // 创建 Decoder
        this.decoder = WhisperTextDecoder.fromConfig(config);

        // 注册子模块
        this.addModule('encoder', this.encoder);
        this.addModule('decoder', this.decoder);
    }

    /**
     * 从模型尺寸创建模型
     */
    static fromSize(size: WhisperModelSize): WhisperModel {
        const config = getWhisperConfig(size);
        return new WhisperModel(config);
    }

    /**
     * 编码音频
     *
     * @param mel - Mel 频谱图，形状 (batch, n_mels, n_frames)
     * @returns 音频特征，形状 (batch, n_frames/2, n_state)
     */
    async embedAudio(mel: Tensor): Promise<Tensor> {
        return await this.encoder.call(mel) as Tensor;
    }

    /**
     * 计算 logits
     *
     * @param tokens - Token IDs，形状 (batch, seq_len)
     * @param audioFeatures - Encoder 输出，形状 (batch, audio_len, n_state)
     * @param options - 可选参数
     * @returns logits，形状 (batch, seq_len, vocab_size)
     */
    async logits(
        tokens: Tensor,
        audioFeatures: Tensor,
        options?: WhisperModelForwardOptions
    ): Promise<Tensor> {
        return await this.decoder.call(tokens, audioFeatures, {
            offset: options?.decoderOffset,
        }) as Tensor;
    }

    /**
     * 前向传播 (完整流程)
     *
     * @param mel - Mel 频谱图，形状 (batch, n_mels, n_frames)
     * @param tokens - Token IDs，形状 (batch, seq_len)
     * @returns logits，形状 (batch, seq_len, vocab_size)
     */
    async forward(mel: Tensor, tokens: Tensor): Promise<Tensor> {
        const audioFeatures = await this.embedAudio(mel);
        return await this.logits(tokens, audioFeatures);
    }

    /**
     * 生成初始 decoder token 序列
     *
     * Whisper 使用特殊的提示格式：
     * <|startoftranscript|><|language|><|task|><|notimestamps|>
     *
     * @param options - 生成选项
     * @returns 初始 token 序列
     */
    getInitialTokens(options?: WhisperGenerateOptions): number[] {
        const {
            language = 'en',
            task = 'transcribe',
            withTimestamps = false,
        } = options ?? {};

        const tokens: number[] = [];

        // <|startoftranscript|>
        tokens.push(WHISPER_SPECIAL_TOKENS.SOT);

        // <|language|>
        const langToken = WHISPER_LANGUAGE_TOKENS[language];
        if (langToken !== undefined) {
            tokens.push(langToken);
        } else {
            // 默认英语
            tokens.push(WHISPER_LANGUAGE_TOKENS.en);
        }

        // <|task|>
        if (task === 'translate') {
            tokens.push(WHISPER_SPECIAL_TOKENS.TRANSLATE);
        } else {
            tokens.push(WHISPER_SPECIAL_TOKENS.TRANSCRIBE);
        }

        // <|notimestamps|> (可选)
        if (!withTimestamps) {
            tokens.push(WHISPER_SPECIAL_TOKENS.NO_TIMESTAMPS);
        }

        return tokens;
    }

    /**
     * 生成下一个 token
     *
     * @param tokens - 当前 token 序列，形状 (batch, seq_len)
     * @param audioFeatures - Encoder 输出
     * @returns 下一个 token ID
     */
    async generateNextToken(
        tokens: Tensor,
        audioFeatures: Tensor
    ): Promise<number> {
        // 获取 logits
        const logits = await this.logits(tokens, audioFeatures);

        // 获取最后一个位置的 logits
        const seqLen = logits.shape[1];
        const lastLogits = select(logits, 1, seqLen - 1);

        // 简单的 argmax 采样
        const argmaxResult = lastLogits.argmax(-1);
        const data = await argmaxResult.dataAsync();
        const nextTokenId = Number(data[0]);

        // 清理
        logits.dispose();
        lastLogits.dispose();
        argmaxResult.dispose();

        return nextTokenId;
    }

    /**
     * 自回归生成文本
     *
     * @param mel - Mel 频谱图，形状 (batch, n_mels, n_frames)
     * @param options - 生成选项
     * @returns 生成的 token IDs
     */
    async generate(
        mel: Tensor,
        options?: WhisperGenerateOptions
    ): Promise<number[]> {
        const {
            maxNewTokens = 224,  // Whisper 默认最大长度的一半
        } = options ?? {};

        // 编码音频
        console.log('[Whisper] 开始编码音频...');
        const startEncode = performance.now();
        const audioFeatures = await this.embedAudio(mel);
        console.log(`[Whisper] 音频编码完成: ${(performance.now() - startEncode).toFixed(0)}ms`);
        console.log(`[Whisper] audioFeatures shape: [${audioFeatures.shape}]`);
        
        // Debug: 打印编码器输出统计
        const afData = await audioFeatures.dataAsync() as Float32Array;
        let afMin = afData[0], afMax = afData[0], afSum = 0;
        for (let i = 0; i < afData.length; i++) {
            const v = afData[i];
            if (v < afMin) afMin = v;
            if (v > afMax) afMax = v;
            afSum += v;
        }
        const afMean = afSum / afData.length;
        console.log(`[Whisper] audioFeatures stats: min=${afMin.toFixed(4)}, max=${afMax.toFixed(4)}, mean=${afMean.toFixed(4)}`);
        console.log(`[Whisper] audioFeatures [0,0,:8]: [${Array.from(afData).slice(0, 8).map(v => v.toFixed(4)).join(', ')}]`);

        // 获取初始 token
        const tokenIds = this.getInitialTokens(options);
        const generatedTokens: number[] = [];
        console.log(`[Whisper] 初始 tokens: [${tokenIds.join(', ')}]`);

        // 自回归生成
        for (let i = 0; i < maxNewTokens; i++) {
            const startStep = performance.now();
            
            // 创建输入张量
            const inputIds = new Tensor(
                new Int32Array(tokenIds),
                { dtype: 'int32', shape: [1, tokenIds.length] }
            );

            // 生成下一个 token
            const nextTokenId = await this.generateNextToken(inputIds, audioFeatures);
            inputIds.dispose();
            
            console.log(`[Whisper] Step ${i + 1}: token=${nextTokenId}, time=${(performance.now() - startStep).toFixed(0)}ms`);

            generatedTokens.push(nextTokenId);

            // 检查是否遇到结束符
            if (nextTokenId === WHISPER_SPECIAL_TOKENS.EOT) {
                console.log('[Whisper] 遇到 EOT，停止生成');
                break;
            }

            // 添加到序列
            tokenIds.push(nextTokenId);

            // 防止超出最大长度
            if (tokenIds.length >= this.config.maxTargetPositions) {
                console.log('[Whisper] 达到最大长度，停止生成');
                break;
            }
        }

        // 清理
        audioFeatures.dispose();

        return generatedTokens;
    }
}
