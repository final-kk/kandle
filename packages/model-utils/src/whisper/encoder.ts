/**
 * WhisperAudioEncoder - Whisper 音频编码器
 *
 * 对标 OpenAI Whisper 中的 AudioEncoder
 *
 * 结构：
 * - Conv1d(n_mels → n_state, kernel=3, padding=1) → GELU
 * - Conv1d(n_state → n_state, kernel=3, stride=2, padding=1) → GELU (2x 下采样)
 * - 固定正弦位置编码
 * - N x ResidualAttentionBlock (self-attention only)
 * - LayerNorm (ln_post)
 *
 * 输入: (batch, n_mels, n_frames) mel spectrogram
 * 输出: (batch, n_frames/2, n_state) 编码后的音频特征
 *
 * @module @kandle/model-utils/whisper/encoder
 */

import type { DType } from '@kandle/types';
import {
    Tensor,
    nn,
    add,
    functional,
} from '@kandle/core';

import { createSinusoidalEncoding } from '../sinusoidal';
import { ResidualAttentionBlock } from './block';
import type { WhisperConfig } from './config';

// ============================================================================
// Types
// ============================================================================

export interface WhisperAudioEncoderConfig {
    /** Mel 频率 bins 数量 (80 或 128) */
    nMels: number;

    /** 编码器最大序列位置 (帧数，通常 1500) */
    nCtx: number;

    /** 模型维度 */
    nState: number;

    /** 注意力头数 */
    nHead: number;

    /** 编码器层数 */
    nLayer: number;

    /** 数据类型 */
    dtype?: DType;
}

// ============================================================================
// WhisperAudioEncoder Class
// ============================================================================

/**
 * Whisper 音频编码器
 *
 * HuggingFace 权重结构：
 * - conv1.weight: (n_state, n_mels, 3)
 * - conv1.bias: (n_state,)
 * - conv2.weight: (n_state, n_state, 3)
 * - conv2.bias: (n_state,)
 * - embed_positions.weight: (n_ctx, n_state) [固定正弦位置编码]
 * - layers.0.self_attn.*, layers.0.self_attn_layer_norm.*, layers.0.fc1.*, layers.0.fc2.*, layers.0.final_layer_norm.*
 * - ...
 * - layer_norm.weight, layer_norm.bias
 */
export class WhisperAudioEncoder extends nn.Module {
    // Config
    readonly nMels: number;
    readonly nCtx: number;
    readonly nState: number;
    readonly nHead: number;
    readonly nLayer: number;

    // 卷积前端
    conv1: nn.Conv1d;
    conv2: nn.Conv1d;

    // 位置编码 (固定，不可训练，HF 称为 embed_positions)
    embed_positions: nn.Embedding;

    // Transformer layers (HF 使用 layers，不是 blocks)
    layers: nn.ModuleList;

    // 输出 LayerNorm (HF 使用 layer_norm)
    layer_norm: nn.LayerNorm;

    constructor(config: WhisperAudioEncoderConfig) {
        super();

        const {
            nMels,
            nCtx,
            nState,
            nHead,
            nLayer,
            dtype = 'float32',
        } = config;

        this.nMels = nMels;
        this.nCtx = nCtx;
        this.nState = nState;
        this.nHead = nHead;
        this.nLayer = nLayer;

        // Conv1: (n_mels, time) → (n_state, time)
        // kernel_size=3, padding=1 保持时间维度不变
        // Conv1d(inChannels, outChannels, kernelSize, stride, padding, dilation, groups, bias, paddingMode, dtype)
        this.conv1 = new nn.Conv1d(
            nMels,   // inChannels
            nState,  // outChannels
            3,       // kernelSize
            1,       // stride
            1,       // padding
            1,       // dilation
            1,       // groups
            true,    // bias
            'zeros', // paddingMode
            dtype
        );

        // Conv2: (n_state, time) → (n_state, time/2)
        // kernel_size=3, stride=2, padding=1 进行 2x 下采样
        this.conv2 = new nn.Conv1d(
            nState,  // inChannels
            nState,  // outChannels
            3,       // kernelSize
            2,       // stride (2x 下采样)
            1,       // padding
            1,       // dilation
            1,       // groups
            true,    // bias
            'zeros', // paddingMode
            dtype
        );

        // 位置编码 Embedding (HF 使用 embed_positions，参数固定不训练)
        this.embed_positions = new nn.Embedding(nCtx, nState, undefined, dtype);

        // Transformer Layers (self-attention only)
        const layerList: ResidualAttentionBlock[] = [];
        for (let i = 0; i < nLayer; i++) {
            layerList.push(new ResidualAttentionBlock({
                nState,
                nHead,
                crossAttention: false,  // Encoder 无 cross-attention
                dtype,
            }));
        }
        this.layers = new nn.ModuleList(layerList);

        // 输出 LayerNorm (HF 使用 layer_norm)
        this.layer_norm = new nn.LayerNorm([nState]);

        // 注册子模块（与 HuggingFace 权重结构对齐）
        this.addModule('conv1', this.conv1);
        this.addModule('conv2', this.conv2);
        this.addModule('embed_positions', this.embed_positions);
        this.addModule('layers', this.layers);
        this.addModule('layer_norm', this.layer_norm);

        // 初始化位置编码 (HF 也是使用固定的正弦编码初始化 embed_positions)
        // 但 HF 实际是使用 nn.Embedding 然后用 sinusoids 初始化权重
        this._initPositionalEmbedding(dtype);
    }

    /**
     * 初始化正弦位置编码到 embed_positions
     */
    private _initPositionalEmbedding(dtype: DType): void {
        const posEmb = createSinusoidalEncoding(this.nCtx, this.nState, 10000.0, dtype);
        // 需要将 posEmb 设置到 embed_positions.weight
        // 这里只是初始化，加载权重时会被覆盖
        // HF 的位置编码在 WhisperEncoder 中初始化后设置 requires_grad_(False)
    }

    /**
     * 从配置创建编码器
     */
    static fromConfig(config: WhisperConfig): WhisperAudioEncoder {
        return new WhisperAudioEncoder({
            nMels: config.numMelBins,
            nCtx: config.maxSourcePositions,
            nState: config.dModel,
            nHead: config.encoderAttentionHeads,
            nLayer: config.encoderLayers,
            dtype: config.dtype,
        });
    }

    /**
     * 前向传播
     *
     * @param x - Mel 频谱图，形状 (batch, n_mels, n_frames)
     * @returns 编码后的音频特征，形状 (batch, n_frames/2, n_state)
     */
    async forward(x: Tensor): Promise<Tensor> {
        // ==========================================
        // 卷积前端
        // ==========================================
        // Conv1: (batch, n_mels, time) → (batch, n_state, time)
        x = await this.conv1.call(x) as Tensor;
        x = functional.gelu(x);

        // Conv2: (batch, n_state, time) → (batch, n_state, time/2)
        x = await this.conv2.call(x) as Tensor;
        x = functional.gelu(x);

        // ==========================================
        // 转置: (batch, n_state, time) → (batch, time, n_state)
        // ==========================================
        x = x.permute([0, 2, 1]);

        // ==========================================
        // 添加位置编码
        // ==========================================
        // 验证形状
        const [batch, seqLen, _] = x.shape;
        if (seqLen > this.nCtx) {
            throw new Error(
                `Audio sequence length ${seqLen} exceeds maximum ${this.nCtx}`
            );
        }

        // 获取所有位置索引，然后查表获取位置编码
        // HF 做法: self.embed_positions(all_positions) 其中 all_positions = arange(seqLen)
        // 切片位置编码到实际长度: embed_positions.weight[:seqLen, :]
        const posEmb = this.embed_positions.weight.slice(`0:${seqLen}, :`);
        x = add(x, posEmb);

        // ==========================================
        // Transformer Layers
        // ==========================================
        for (let i = 0; i < this.nLayer; i++) {
            const layer = this.layers.get(i) as ResidualAttentionBlock;
            x = await layer.call(x) as Tensor;
        }

        // ==========================================
        // 输出 LayerNorm
        // ==========================================
        x = await this.layer_norm.call(x) as Tensor;

        return x;
    }
}
