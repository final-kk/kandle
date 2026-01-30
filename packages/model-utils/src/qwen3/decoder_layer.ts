/**
 * Qwen3DecoderLayer - Qwen3 Transformer 解码器层
 *
 * 对标 HuggingFace Transformers 中的 Qwen3DecoderLayer
 *
 * 结构：
 * - Self-Attention with GQA (Grouped Query Attention)
 * - Pre-Normalization with RMSNorm
 * - SwiGLU MLP
 * - Residual connections
 *
 * @example
 * ```ts
 * const layer = new Qwen3DecoderLayer({
 *     hiddenSize: 1024,
 *     intermediateSize: 3072,
 *     numAttentionHeads: 16,
 *     numKeyValueHeads: 8,
 *     rmsNormEps: 1e-6,
 * });
 *
 * const output = await layer.call(hidden_states, { positionEmbeddings });
 * ```
 *
 * @module @kandle/model-utils/qwen3/decoder_layer
 */

import type { DType } from '@kandle/types';
import {
    Tensor,
    nn,
} from '@kandle/core';

import { GroupedQueryAttention, type GroupedQueryAttentionForwardOptions } from '../attention';
import { SwiGLUMLP } from '../mlp';

// ============================================================================
// Types
// ============================================================================

/**
 * Qwen3DecoderLayer 配置
 */
export interface Qwen3DecoderLayerConfig {
    /** 隐藏层维度 */
    hiddenSize: number;

    /** FFN 中间维度 */
    intermediateSize: number;

    /** 注意力头数 (Query) */
    numAttentionHeads: number;

    /** KV 头数 (用于 GQA) */
    numKeyValueHeads: number;

    /** 每个头的维度 (默认 hiddenSize / numAttentionHeads) */
    headDim?: number;

    /** RMSNorm epsilon */
    rmsNormEps?: number;

    /** 注意力偏置 */
    attentionBias?: boolean;

    /** MLP 偏置 */
    mlpBias?: boolean;

    /**
     * 是否对 Q/K 应用 RMSNorm (Qwen3 默认启用)
     * @default true
     */
    qkNorm?: boolean;

    /** 数据类型 */
    dtype?: DType;
}

/**
 * Forward 选项
 */
export interface Qwen3DecoderLayerForwardOptions {
    /** 位置嵌入 (cos, sin) 用于 RoPE */
    positionEmbeddings?: {
        cos: Tensor;
        sin: Tensor;
    };

    /** 注意力掩码 */
    attnMask?: Tensor;

    /** 是否使用因果掩码 */
    isCausal?: boolean;

    /** KV Cache 更新回调 (从 GQA 透传) */
    kvCacheUpdateFn?: (key: Tensor, value: Tensor, startPos: number) => [Tensor, Tensor];

    /** 缓存起始位置 (从 GQA 透传) */
    cachePosition?: number;

    /**
     * 是否捕获 attention weights (透传到 GQA)
     *
     * 开启后可通过 self_attn.lastAttentionWeights 获取
     * 形状: [batch, numHeads, querySeqLen, keySeqLen]
     *
     * @default false
     */
    captureAttentionWeights?: boolean;
}

// ============================================================================
// Qwen3DecoderLayer Class
// ============================================================================

/**
 * Qwen3 Transformer 解码器层
 *
 * 权重结构（适配 HuggingFace 模型）：
 * - self_attn.q_proj.weight, self_attn.k_proj.weight, self_attn.v_proj.weight, self_attn.o_proj.weight
 * - mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
 * - input_layernorm.weight
 * - post_attention_layernorm.weight
 */
export class Qwen3DecoderLayer extends nn.Module {
    // Config
    readonly hiddenSize: number;

    // 子模块
    self_attn: GroupedQueryAttention;
    mlp: SwiGLUMLP;
    input_layernorm: nn.RMSNorm;
    post_attention_layernorm: nn.RMSNorm;

    constructor(config: Qwen3DecoderLayerConfig) {
        super();

        const {
            hiddenSize,
            intermediateSize,
            numAttentionHeads,
            numKeyValueHeads,
            headDim = Math.floor(hiddenSize / numAttentionHeads),
            rmsNormEps = 1e-6,
            attentionBias = true,
            mlpBias = false,
            qkNorm = true,  // Qwen3 默认启用 Q/K Normalization
            dtype = 'float32',
        } = config;

        this.hiddenSize = hiddenSize;

        // Self-Attention (GQA)
        this.self_attn = new GroupedQueryAttention({
            embedDim: hiddenSize,
            numHeads: numAttentionHeads,
            numKvHeads: numKeyValueHeads,
            headDim,
            bias: attentionBias,
            qkNorm,       // Qwen3 特有
            rmsNormEps,   // 传递给 Q/K Norm
            dtype,
        });

        // MLP (SwiGLU)
        this.mlp = new SwiGLUMLP({
            hiddenSize,
            intermediateSize,
            bias: mlpBias,
            dtype,
        });

        // Pre-Normalization layers
        this.input_layernorm = new nn.RMSNorm([hiddenSize], rmsNormEps);
        this.post_attention_layernorm = new nn.RMSNorm([hiddenSize], rmsNormEps);

        // 注册子模块
        this.addModule('self_attn', this.self_attn);
        this.addModule('mlp', this.mlp);
        this.addModule('input_layernorm', this.input_layernorm);
        this.addModule('post_attention_layernorm', this.post_attention_layernorm);
    }

    /**
     * 前向传播
     *
     * @param hiddenStates 输入张量，形状 (batch, seq_len, hidden_size)
     * @param options 可选参数
     * @returns 输出张量，形状 (batch, seq_len, hidden_size)
     */
    async forward(
        hiddenStates: Tensor,
        options?: Qwen3DecoderLayerForwardOptions
    ): Promise<Tensor> {
        const {
            positionEmbeddings,
            attnMask,
            isCausal = true,
            kvCacheUpdateFn,
            cachePosition,
            captureAttentionWeights = false,
        } = options ?? {};

        // ==========================================
        // Self-Attention Block (Pre-Norm)
        // ==========================================
        const residual = hiddenStates;

        // Pre-LN
        const normed1 = await this.input_layernorm.call(hiddenStates) as Tensor;

        // Self-Attention (传递 KV Cache 参数)
        const attnOptions: GroupedQueryAttentionForwardOptions = {
            positionEmbeddings,
            attnMask,
            isCausal,
            kvCacheUpdateFn,
            cachePosition,
            captureAttentionWeights,
        };
        const attnOutput = await this.self_attn.call(normed1, attnOptions) as Tensor;

        // Residual connection
        const afterAttn = residual.add(attnOutput);

        // ==========================================
        // MLP Block (Pre-Norm)
        // ==========================================
        const residual2 = afterAttn;

        // Pre-LN
        const normed2 = await this.post_attention_layernorm.call(afterAttn) as Tensor;

        // MLP
        const mlpOutput = await this.mlp.call(normed2) as Tensor;

        // Residual connection
        const output = residual2.add(mlpOutput);

        return output;
    }

    /**
     * 额外表示信息
     */
    protected extraRepr(): string {
        return `hiddenSize=${this.hiddenSize}`;
    }
}
