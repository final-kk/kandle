/**
 * Qwen3Model - Qwen3 Transformer 模型 (不含语言模型头)
 *
 * 对标 HuggingFace Transformers 中的 Qwen3Model
 *
 * 结构：
 * - Token Embedding
 * - N x Qwen3DecoderLayer
 * - Final RMSNorm
 * - RotaryEmbedding (用于位置编码)
 *
 * @example
 * ```ts
 * const model = new Qwen3Model({
 *     vocabSize: 151669,
 *     hiddenSize: 1024,
 *     intermediateSize: 3072,
 *     numHiddenLayers: 28,
 *     numAttentionHeads: 16,
 *     numKeyValueHeads: 8,
 * });
 *
 * const hidden_states = await model.call(input_ids);
 * ```
 *
 * @module @kandle/model-utils/qwen3/model
 */

import type { DType } from "@kandle/types";
import { Tensor, nn, tidy } from "@kandle/core";

import { RotaryEmbedding } from "../rope";
import { Qwen3DecoderLayer, type Qwen3DecoderLayerConfig } from "./decoder_layer";
import { StaticKVCache } from "../kv-cache";

// ============================================================================
// Types
// ============================================================================

/**
 * Qwen3Model 配置
 *
 * 对应 HuggingFace Qwen3Config
 */
export interface Qwen3ModelConfig {
    /** 词汇表大小 */
    vocabSize: number;

    /** 隐藏层维度 */
    hiddenSize: number;

    /** FFN 中间维度 */
    intermediateSize: number;

    /** Transformer 层数 */
    numHiddenLayers: number;

    /** 注意力头数 (Query) */
    numAttentionHeads: number;

    /** KV 头数 (用于 GQA) */
    numKeyValueHeads: number;

    /** 每个头的维度 (默认 hiddenSize / numAttentionHeads) */
    headDim?: number;

    /** 最大序列长度 */
    maxPositionEmbeddings?: number;

    /** RoPE base (theta) */
    ropeTheta?: number;

    /** RMSNorm epsilon */
    rmsNormEps?: number;

    /** 注意力偏置 */
    attentionBias?: boolean;

    /** MLP 偏置 */
    mlpBias?: boolean;

    /** 数据类型 */
    dtype?: DType;
}

/**
 * Forward 选项
 */
export interface Qwen3ModelForwardOptions {
    /** 注意力掩码 */
    attnMask?: Tensor;

    /** 是否使用因果掩码 (默认 true) */
    isCausal?: boolean;

    /**
     * KV Cache 实例
     *
     * 如果提供，将在每层缓存 K/V 并从缓存中读取历史 K/V
     */
    kvCache?: StaticKVCache;

    /**
     * 缓存起始位置
     *
     * - Prefill 阶段: 0
     * - Decode 阶段: 已处理的 token 数
     */
    cachePosition?: number;

    /**
     * Logit Lens: 需要收集 hidden states 的层索引
     *
     * 如果提供，将在这些层收集 hidden states（已应用该层后的输出）
     * 用于 Logit Lens 可视化
     *
     * @example [0, 7, 14, 21, 27] // 收集第 0, 7, 14, 21, 27 层的输出
     */
    collectLayerIndices?: number[];
}

/**
 * Forward 结果（带 Logit Lens）
 */
export interface Qwen3ModelForwardResult {
    /** 最终 hidden states (经过 final norm) */
    hiddenStates: Tensor;

    /**
     * 各层的 hidden states（经过该层后，但未应用 final norm）
     * key 为层索引
     */
    layerHiddenStates?: Map<number, Tensor>;
}

// ============================================================================
// Qwen3Model Class
// ============================================================================

/**
 * Qwen3 Transformer 模型
 *
 * 权重结构（适配 HuggingFace 模型）：
 * - embed_tokens.weight: [vocab_size, hidden_size]
 * - layers.0.self_attn.q_proj.weight, ...
 * - layers.0.mlp.gate_proj.weight, ...
 * - layers.0.input_layernorm.weight, ...
 * - ...
 * - norm.weight: [hidden_size]
 */
export class Qwen3Model extends nn.Module {
    // Config
    readonly config: Qwen3ModelConfig;

    // 子模块
    embed_tokens: nn.Embedding;
    layers: nn.ModuleList;
    norm: nn.RMSNorm;
    rotary_emb: RotaryEmbedding;

    constructor(config: Qwen3ModelConfig) {
        super();

        const {
            vocabSize,
            hiddenSize,
            intermediateSize,
            numHiddenLayers,
            numAttentionHeads,
            numKeyValueHeads,
            headDim = Math.floor(hiddenSize / numAttentionHeads),
            maxPositionEmbeddings = 32768,
            ropeTheta = 1000000, // Qwen3 使用较大的 rope_theta
            rmsNormEps = 1e-6,
            attentionBias = true,
            mlpBias = false,
            dtype = "float32",
        } = config;

        this.config = config;

        // Token Embedding
        this.embed_tokens = new nn.Embedding({
            numEmbeddings: vocabSize,
            embeddingDim: hiddenSize,
        });

        // Decoder Layers
        const layerConfig: Qwen3DecoderLayerConfig = {
            hiddenSize,
            intermediateSize,
            numAttentionHeads,
            numKeyValueHeads,
            headDim,
            rmsNormEps,
            attentionBias,
            mlpBias,
            dtype,
        };

        const layers: Qwen3DecoderLayer[] = [];
        for (let i = 0; i < numHiddenLayers; i++) {
            layers.push(new Qwen3DecoderLayer(layerConfig));
        }
        this.layers = new nn.ModuleList(layers);

        // Final Normalization
        this.norm = new nn.RMSNorm([hiddenSize], rmsNormEps);

        // Rotary Position Embedding
        this.rotary_emb = new RotaryEmbedding({
            dim: headDim,
            maxSeqLen: maxPositionEmbeddings,
            base: ropeTheta,
            dtype,
        });

        // 注册子模块
        this.addModule("embed_tokens", this.embed_tokens);
        this.addModule("layers", this.layers);
        this.addModule("norm", this.norm);
        // rotary_emb 作为 buffer 不需要注册为子模块 (它没有可学习参数)
    }

    /**
     * 前向传播
     *
     * @param inputIds 输入 token IDs，形状 (batch, seq_len)
     * @param options 可选参数
     * @returns 隐藏状态，形状 (batch, seq_len, hidden_size)
     */
    async forward(inputIds: Tensor, options?: Qwen3ModelForwardOptions): Promise<Tensor> {
        const result = await this.forwardWithLayerOutputs(inputIds, options);
        return result.hiddenStates;
    }

    /**
     * 带层输出收集的前向传播
     *
     * 用于 Logit Lens 等可解释性功能
     *
     * @param inputIds 输入 token IDs，形状 (batch, seq_len)
     * @param options 可选参数，包含 collectLayerIndices
     * @returns 最终 hidden states 和各层 hidden states
     */
    async forwardWithLayerOutputs(
        inputIds: Tensor,
        options?: Qwen3ModelForwardOptions
    ): Promise<Qwen3ModelForwardResult> {
        const {
            attnMask,
            isCausal = true,
            kvCache,
            cachePosition = 0,
            collectLayerIndices,
        } = options ?? {};

        const seqLen = inputIds.shape[1];
        const collectSet = collectLayerIndices ? new Set(collectLayerIndices) : null;
        const layerHiddenStates = collectSet ? new Map<number, Tensor>() : undefined;

        // Token Embedding
        let hiddenStates = (await this.embed_tokens.call(inputIds)) as Tensor;

        // 生成位置嵌入 (RoPE) - 使用同步 tidy 包装
        // positionIds 会被释放，positionEmbeddings (cos/sin) 会被保留
        const positionEmbeddings = tidy(() => {
            const positionIds = this.rotary_emb.getPositionIds(seqLen, cachePosition);
            return this.rotary_emb.forward(positionIds);
        });

        // Decoder Layers
        let layerIdx = 0;
        for (const layer of this.layers) {
            const decoderLayer = layer as Qwen3DecoderLayer;

            // 如果提供了 KV Cache，为当前层创建更新回调
            let kvCacheUpdateFn:
                | ((k: Tensor, v: Tensor, pos: number) => [Tensor, Tensor])
                | undefined;
            if (kvCache) {
                const currentLayerIdx = layerIdx; // 捕获当前层索引
                kvCacheUpdateFn = (k: Tensor, v: Tensor, pos: number): [Tensor, Tensor] => {
                    return kvCache.update(currentLayerIdx, k, v, pos);
                };
            }

            hiddenStates = (await decoderLayer.call(hiddenStates, {
                positionEmbeddings,
                attnMask,
                isCausal,
                kvCacheUpdateFn,
                cachePosition,
            })) as Tensor;

            // 如果需要收集该层的输出，保存一份副本
            if (collectSet && collectSet.has(layerIdx) && layerHiddenStates) {
                // 对于 Logit Lens，我们需要保存该层输出的副本
                // 因为 hiddenStates 会在后续层被覆盖
                layerHiddenStates.set(layerIdx, hiddenStates.clone());
            }

            layerIdx++;
        }

        // Final LayerNorm
        hiddenStates = (await this.norm.call(hiddenStates)) as Tensor;

        return {
            hiddenStates,
            layerHiddenStates,
        };
    }

    /**
     * 额外表示信息
     */
    protected extraRepr(): string {
        return `vocabSize=${this.config.vocabSize}, hiddenSize=${this.config.hiddenSize}, numLayers=${this.config.numHiddenLayers}`;
    }
}
