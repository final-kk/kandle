/**
 * WhisperTextDecoder - Whisper 文本解码器
 *
 * 对标 OpenAI Whisper 中的 TextDecoder
 *
 * 结构：
 * - Token Embedding
 * - 可学习位置编码 (nn.Parameter)
 * - N x ResidualAttentionBlock (self-attention + cross-attention)
 * - LayerNorm (ln)
 * - 输出投影 (与 token embedding 共享权重)
 *
 * 输入:
 * - tokens: (batch, seq_len) token IDs
 * - xa: (batch, audio_len, n_state) encoder 输出
 *
 * 输出: (batch, seq_len, vocab_size) logits
 *
 * @module @kandle/model-utils/whisper/decoder
 */

import type { DType } from '@kandle/types';
import {
    Tensor,
    nn,
    add,
    matmul,
    transpose,
    tril,
    ones,
    full,
    slice,
} from '@kandle/core';

import { ResidualAttentionBlock } from './block';
import type { WhisperConfig } from './config';

// ============================================================================
// Types
// ============================================================================

export interface WhisperTextDecoderConfig {
    /** 词汇表大小 */
    nVocab: number;

    /** 最大序列长度 */
    nCtx: number;

    /** 模型维度 */
    nState: number;

    /** 注意力头数 */
    nHead: number;

    /** 解码器层数 */
    nLayer: number;

    /** 数据类型 */
    dtype?: DType;
}

export interface WhisperTextDecoderForwardOptions {
    /** 缓存偏移量 (用于增量解码) */
    offset?: number;
}

// ============================================================================
// WhisperTextDecoder Class
// ============================================================================

/**
 * Whisper 文本解码器
 *
 * HuggingFace 权重结构：
 * - embed_tokens.weight: (vocab_size, n_state)
 * - embed_positions.weight: (n_ctx, n_state) [可学习位置编码]
 * - layers.0.self_attn.*, layers.0.self_attn_layer_norm.*, layers.0.encoder_attn.*, layers.0.encoder_attn_layer_norm.*, layers.0.fc1.*, layers.0.fc2.*, layers.0.final_layer_norm.*
 * - ...
 * - layer_norm.weight, layer_norm.bias
 * - (WhisperForConditionalGeneration 有 proj_out，与 embed_tokens 共享)
 */
export class WhisperTextDecoder extends nn.Module {
    // Config
    readonly nVocab: number;
    readonly nCtx: number;
    readonly nState: number;
    readonly nHead: number;
    readonly nLayer: number;

    // Token Embedding (HF 使用 embed_tokens)
    embed_tokens: nn.Embedding;

    // 可学习位置编码 (HF 使用 embed_positions，是 nn.Embedding)
    embed_positions: nn.Embedding;

    // Transformer layers (HF 使用 layers，带 cross-attention)
    layers: nn.ModuleList;

    // 输出 LayerNorm (HF 使用 layer_norm)
    layer_norm: nn.LayerNorm;

    // 因果掩码 (预计算)
    private _mask: Tensor | null = null;

    constructor(config: WhisperTextDecoderConfig) {
        super();

        const {
            nVocab,
            nCtx,
            nState,
            nHead,
            nLayer,
            dtype = 'float32',
        } = config;

        this.nVocab = nVocab;
        this.nCtx = nCtx;
        this.nState = nState;
        this.nHead = nHead;
        this.nLayer = nLayer;

        // Token Embedding (HF: embed_tokens)
        this.embed_tokens = new nn.Embedding({
            numEmbeddings: nVocab,
            embeddingDim: nState,
            dtype,
        });

        // 可学习位置编码 (HF: embed_positions，是 WhisperPositionalEmbedding 继承自 nn.Embedding)
        this.embed_positions = new nn.Embedding({
            numEmbeddings: nCtx,
            embeddingDim: nState,
            dtype,
        });

        // Transformer Layers (with cross-attention)
        const layerList: ResidualAttentionBlock[] = [];
        for (let i = 0; i < nLayer; i++) {
            layerList.push(new ResidualAttentionBlock({
                nState,
                nHead,
                crossAttention: true,  // Decoder 需要 cross-attention
                dtype,
            }));
        }
        this.layers = new nn.ModuleList(layerList);

        // 输出 LayerNorm (HF: layer_norm)
        this.layer_norm = new nn.LayerNorm([nState]);

        // 预计算因果掩码
        // mask[i, j] = -inf if j > i else 0
        this._createCausalMask(nCtx, dtype);

        // 注册子模块（与 HuggingFace 权重结构对齐）
        this.addModule('embed_tokens', this.embed_tokens);
        this.addModule('embed_positions', this.embed_positions);
        this.addModule('layers', this.layers);
        this.addModule('layer_norm', this.layer_norm);
    }

    /**
     * 创建因果掩码
     *
     * mask[i, j] = -inf if j > i else 0
     * 用于屏蔽未来 token
     */
    private _createCausalMask(nCtx: number, dtype: DType): void {
        // 创建上三角矩阵，对角线以上为 -inf
        const mask = new Float32Array(nCtx * nCtx);
        for (let i = 0; i < nCtx; i++) {
            for (let j = 0; j < nCtx; j++) {
                if (j > i) {
                    mask[i * nCtx + j] = -Infinity;
                }
            }
        }
        this._mask = new Tensor(mask, { shape: [nCtx, nCtx], dtype });
        this.registerBuffer('mask', this._mask, false);  // 不持久化
    }

    /**
     * 获取因果掩码
     */
    get mask(): Tensor {
        return this._mask!;
    }

    /**
     * 从配置创建解码器
     */
    static fromConfig(config: WhisperConfig): WhisperTextDecoder {
        return new WhisperTextDecoder({
            nVocab: config.vocabSize,
            nCtx: config.maxTargetPositions,
            nState: config.dModel,
            nHead: config.decoderAttentionHeads,
            nLayer: config.decoderLayers,
            dtype: config.dtype,
        });
    }

    /**
     * 前向传播
     *
     * @param tokens - Token IDs，形状 (batch, seq_len)
     * @param xa - Encoder 输出，形状 (batch, audio_len, n_state)
     * @param options - 可选参数
     * @returns logits，形状 (batch, seq_len, vocab_size)
     */
    async forward(
        tokens: Tensor,
        xa: Tensor,
        options?: WhisperTextDecoderForwardOptions
    ): Promise<Tensor> {
        const { offset = 0 } = options ?? {};

        const seqLen = tokens.shape[1];

        // ==========================================
        // Token Embedding + Position Embedding
        // ==========================================
        // Token embedding: (batch, seq_len) → (batch, seq_len, n_state)
        let x = await this.embed_tokens.call(tokens) as Tensor;

        // Position embedding: 获取 [offset : offset + seq_len] 的位置编码
        // HF 使用 embed_positions.weight[offset : offset + seq_len, :]
        const posEmb = this.embed_positions.weight.slice(`${offset}:${offset + seqLen}, :`);
        x = add(x, posEmb);

        // ==========================================
        // Transformer Layers
        // ==========================================
        // 获取因果掩码切片
        // mask[0:seq_len, 0:seq_len]
        const mask = this._mask!.slice(`0:${seqLen}, 0:${seqLen}`);

        for (let i = 0; i < this.nLayer; i++) {
            const layer = this.layers.get(i) as ResidualAttentionBlock;
            x = await layer.call(x, { xa, mask }) as Tensor;
        }

        // ==========================================
        // 输出 LayerNorm
        // ==========================================
        x = await this.layer_norm.call(x) as Tensor;

        // ==========================================
        // 输出投影 (与 embed_tokens 共享权重)
        // ==========================================
        // HF: proj_out 与 decoder.embed_tokens 权重绑定
        // logits = x @ embed_tokens.weight.T
        // x: (batch, seq_len, n_state)
        // weight: (vocab_size, n_state)
        // output: (batch, seq_len, vocab_size)
        const weight = this.embed_tokens.weight;
        const logits = matmul(x, transpose(weight, 0, 1));

        return logits;
    }
}
