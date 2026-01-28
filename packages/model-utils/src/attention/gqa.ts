/**
 * GroupedQueryAttention - 分组查询注意力
 *
 * 支持 MHA、GQA、MQA 统一接口：
 * - numKvHeads = numHeads → 标准 Multi-Head Attention (MHA)
 * - numKvHeads = 1 → Multi-Query Attention (MQA)
 * - 1 < numKvHeads < numHeads → Grouped Query Attention (GQA)
 *
 * 对标 HuggingFace Transformers 中的 LlamaAttention / Qwen3Attention
 *
 * 主要特点：
 * - 使用分开的 Q/K/V 投影层（适合加载 HuggingFace 模型权重）
 * - 默认 batch_first=true
 * - 使用 expand + reshape 实现 KV 头扩展（零拷贝）
 * - 推理优化，移除 dropout
 *
 * @example
 * ```ts
 * // Qwen3-0.6B: 16 query heads, 8 KV heads
 * const attn = new GroupedQueryAttention({
 *     embedDim: 1024,
 *     numHeads: 16,
 *     numKvHeads: 8,
 * });
 *
 * const output = await attn.call(hidden_states);
 * ```
 *
 * @module @kandle/model-utils/attention/gqa
 */

import type { DType } from "@kandle/types";
import {
    Tensor,
    nn,
    view,
    expand,
    contiguous,
    reshape,
    functional,
    matmul,
    softmax,
    add,
    div,
    mul,
} from "@kandle/core";

import { applyRotaryPosEmbDirect } from "../rope";

// ============================================================================
// Types
// ============================================================================

/**
 * GroupedQueryAttention 构造选项
 */
export interface GroupedQueryAttentionOptions {
    /** 模型总维度 (必需) */
    embedDim: number;

    /** Query 头数量 (必需) */
    numHeads: number;

    /** Key/Value 头数量 (默认 = numHeads，即标准 MHA) */
    numKvHeads?: number;

    /** 每个头的维度 (默认 = embedDim / numHeads) */
    headDim?: number;

    /** 是否为投影层添加偏置 (默认 true) */
    bias?: boolean;

    /**
     * 是否对 Q/K 应用 RMSNorm (Qwen3 特有)
     * 默认 false
     */
    qkNorm?: boolean;

    /**
     * RMSNorm epsilon (仅当 qkNorm=true 时使用)
     * 默认 1e-6
     */
    rmsNormEps?: number;

    /** 数据类型 */
    dtype?: DType;
}

/**
 * Forward 选项
 */
export interface GroupedQueryAttentionForwardOptions {
    /** 注意力掩码 (additive, float) */
    attnMask?: Tensor;

    /** 是否使用因果掩码 */
    isCausal?: boolean;

    /** 位置信息 - cos/sin 用于 RoPE (可选) */
    positionEmbeddings?: {
        cos: Tensor;
        sin: Tensor;
    };

    /**
     * KV Cache 更新回调
     *
     * 如果提供，在计算完 K/V 后调用此函数，传入 (key, value, startPos)。
     * 回调应返回更新后的完整 K/V 缓存 [fullK, fullV]。
     *
     * 这种设计允许 Attention 与 KVCache 解耦，
     * 由上层 (如 Qwen3Model) 管理缓存生命周期。
     */
    kvCacheUpdateFn?: (key: Tensor, value: Tensor, startPos: number) => [Tensor, Tensor];

    /**
     * 缓存起始位置
     *
     * 当使用 KV Cache 时，指定当前 K/V 应写入的起始位置。
     * - Prefill: startPos = 0
     * - Decode: startPos = 已处理的 token 数
     */
    cachePosition?: number;

    /**
     * 是否捕获 attention weights
     *
     * 警告：开启此选项会使用 naive attention 实现（非 FlashAttention），
     * 会显著降低性能并增加显存占用。仅用于可视化/调试目的。
     *
     * 捕获的权重可通过 `lastAttentionWeights` 属性获取。
     *
     * @default false
     */
    captureAttentionWeights?: boolean;
}

// ============================================================================
// GroupedQueryAttention Class
// ============================================================================

/**
 * GroupedQueryAttention - 分组查询注意力模块
 *
 * 权重结构（适配 HuggingFace 模型）：
 * - q_proj.weight: [numHeads * headDim, embedDim]
 * - q_proj.bias: [numHeads * headDim]
 * - k_proj.weight: [numKvHeads * headDim, embedDim]
 * - k_proj.bias: [numKvHeads * headDim]
 * - v_proj.weight: [numKvHeads * headDim, embedDim]
 * - v_proj.bias: [numKvHeads * headDim]
 * - o_proj.weight: [embedDim, numHeads * headDim]
 * - o_proj.bias: [embedDim]
 */
export class GroupedQueryAttention extends nn.Module {
    // Config
    readonly embedDim: number;
    readonly numHeads: number; // Query heads
    readonly numKvHeads: number; // Key/Value heads
    readonly headDim: number;
    readonly numKvGroups: number; // numHeads / numKvHeads
    readonly useQkNorm: boolean; // Qwen3 特有的 Q/K normalization

    // 投影层 (作为子模块，便于 state_dict 加载)
    q_proj: nn.Linear;
    k_proj: nn.Linear;
    v_proj: nn.Linear;
    o_proj: nn.Linear;

    // Q/K Normalization (Qwen3 特有, 可选)
    q_norm: nn.RMSNorm | null = null;
    k_norm: nn.RMSNorm | null = null;

    /**
     * 最近一次计算的 attention weights
     *
     * 仅当 forward 调用时 `captureAttentionWeights=true` 时有效。
     * 形状: [batch, numHeads, querySeqLen, keySeqLen]
     *
     * 注意：这是一个 Tensor 引用，调用者需要负责 dispose。
     * 每次 forward 调用会覆盖前一次的值。
     */
    lastAttentionWeights: Tensor | null = null;

    constructor(options: GroupedQueryAttentionOptions) {
        super();

        const {
            embedDim,
            numHeads,
            numKvHeads = numHeads, // 默认 MHA
            headDim = Math.floor(embedDim / numHeads),
            bias = true,
            qkNorm = false,
            rmsNormEps = 1e-6,
            dtype = "float32",
        } = options;

        // 验证参数
        if (numHeads % numKvHeads !== 0) {
            throw new Error(
                `numHeads (${numHeads}) must be divisible by numKvHeads (${numKvHeads})`
            );
        }

        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.numKvGroups = Math.floor(numHeads / numKvHeads);
        this.useQkNorm = qkNorm;

        // Q 投影: embedDim -> numHeads * headDim
        this.q_proj = new nn.Linear(embedDim, numHeads * headDim, bias, dtype);

        // K/V 投影: embedDim -> numKvHeads * headDim (reduced for GQA)
        this.k_proj = new nn.Linear(embedDim, numKvHeads * headDim, bias, dtype);
        this.v_proj = new nn.Linear(embedDim, numKvHeads * headDim, bias, dtype);

        // 输出投影: numHeads * headDim -> embedDim
        this.o_proj = new nn.Linear(numHeads * headDim, embedDim, bias, dtype);

        // Q/K Normalization (Qwen3 特有)
        if (qkNorm) {
            // 注意: normalize 只在 head_dim 上，不是完整的 embed_dim
            this.q_norm = new nn.RMSNorm([headDim], rmsNormEps);
            this.k_norm = new nn.RMSNorm([headDim], rmsNormEps);
            this.addModule("q_norm", this.q_norm);
            this.addModule("k_norm", this.k_norm);
        }

        // 注册子模块
        this.addModule("q_proj", this.q_proj);
        this.addModule("k_proj", this.k_proj);
        this.addModule("v_proj", this.v_proj);
        this.addModule("o_proj", this.o_proj);
    }

    /**
     * 前向传播 (Self-Attention)
     *
     * @param hiddenStates 输入张量，形状 (batch, seq_len, embed_dim)
     * @param options 可选参数
     * @returns 输出张量，形状 (batch, seq_len, embed_dim)
     */
    async forward(
        hiddenStates: Tensor,
        options?: GroupedQueryAttentionForwardOptions
    ): Promise<Tensor> {
        const {
            attnMask,
            isCausal = false,
            positionEmbeddings,
            kvCacheUpdateFn,
            cachePosition = 0,
            captureAttentionWeights = false,
        } = options ?? {};

        // 检查 is_causal 与 attn_mask 互斥
        if (isCausal && attnMask !== undefined) {
            throw new Error("isCausal=true cannot be used with attnMask");
        }

        // 获取维度
        const bsz = hiddenStates.shape[0];
        const seqLen = hiddenStates.shape[1];

        // 线性投影
        const qProj = (await this.q_proj.call(hiddenStates)) as Tensor;
        const kProj = (await this.k_proj.call(hiddenStates)) as Tensor;
        const vProj = (await this.v_proj.call(hiddenStates)) as Tensor;

        // 重塑为多头 (BSHD 格式) - view 不需要 dispose，共享底层存储
        let q = this._reshapeForMultihead(qProj, bsz, seqLen, this.numHeads);
        let k = this._reshapeForMultihead(kProj, bsz, seqLen, this.numKvHeads);
        let v = this._reshapeForMultihead(vProj, bsz, seqLen, this.numKvHeads);

        // Q/K Normalization (Qwen3 特有)
        if (this.useQkNorm && this.q_norm && this.k_norm) {
            const qBeforeNorm = q;
            const kBeforeNorm = k;
            q = (await this.q_norm.call(q)) as Tensor;
            k = (await this.k_norm.call(k)) as Tensor;
            // 注意: qBeforeNorm/kBeforeNorm 是 view，不能直接 dispose
            // 但 norm 输出是新张量，qProj/kProj 可以释放
        }
        // 释放投影输出 (norm 已创建新张量，或者没有 norm 时 q/k/v 是 view)
        // 注意：即使 q/k/v 是 view 共享 storage，也需要 dispose 原始投影输出
        qProj.dispose();
        kProj.dispose();
        vProj.dispose();

        // Permute 到 BHSD 格式 - 创建新的 contiguous 张量
        const qBeforePermute = q;
        const kBeforePermute = k;
        const vBeforePermute = v;
        q = this._permuteToHeadsFirst(q);
        k = this._permuteToHeadsFirst(k);
        v = this._permuteToHeadsFirst(v);
        // ⚠️ 关键修复：permute+contiguous 创建新张量，总是释放旧的 view/tensor
        // 无论是否有 QK Norm，_reshapeForMultihead 产生的 view 都需要 dispose
        qBeforePermute.dispose();
        kBeforePermute.dispose();
        vBeforePermute.dispose();

        // 应用 RoPE (如果提供)
        if (positionEmbeddings) {
            const { cos, sin } = positionEmbeddings;
            [q, k] = this._applyRotaryEmbQK(q, k, cos, sin);
        }

        // ==========================================
        // KV Cache 更新 (如果提供了回调)
        // ==========================================
        let cachedK = k;
        let cachedV = v;
        let kvSeqLen = seqLen;

        if (kvCacheUpdateFn) {
            [cachedK, cachedV] = kvCacheUpdateFn(k, v, cachePosition);
            kvSeqLen = cachedK.shape[2];
        }

        // GQA: 扩展 K/V 头以匹配 Q 头数量
        let expandedK = cachedK;
        let expandedV = cachedV;
        if (this.numKvGroups > 1) {
            expandedK = this._repeatKv(cachedK, bsz, kvSeqLen);
            expandedV = this._repeatKv(cachedV, bsz, kvSeqLen);
        }

        // KV Cache decode 阶段处理
        let effectiveIsCausal = isCausal;
        if (kvCacheUpdateFn && seqLen === 1 && kvSeqLen > 1) {
            effectiveIsCausal = false;
        }

        // ==========================================
        // Scaled Dot-Product Attention
        // ==========================================
        let attnOutput: Tensor;

        if (captureAttentionWeights) {
            // Naive attention 实现 - 用于捕获 attention weights
            // 警告：性能较差，仅用于可视化
            const scale = 1.0 / Math.sqrt(this.headDim);

            // Step 1: QK^T * scale
            // q: [batch, heads, q_seq, head_dim]
            // expandedK: [batch, heads, kv_seq, head_dim]
            // 需要 K 转置: [batch, heads, head_dim, kv_seq]
            const kT = expandedK.permute([0, 1, 3, 2]);
            const kTContiguous = contiguous(kT);
            kT.dispose();

            // attnScores: [batch, heads, q_seq, kv_seq]
            const attnScores = matmul(q, kTContiguous);
            kTContiguous.dispose();

            // 应用 scale
            const scaledScores = mul(attnScores, scale);
            attnScores.dispose();

            // Step 2: 应用因果掩码 (如果需要)
            let maskedScores = scaledScores;
            if (effectiveIsCausal && seqLen > 1) {
                // 创建因果掩码: 上三角为 -Infinity
                // 对于 decode 阶段 (seqLen=1)，不需要掩码
                const maskData = new Float32Array(seqLen * kvSeqLen);
                for (let i = 0; i < seqLen; i++) {
                    for (let j = 0; j < kvSeqLen; j++) {
                        // 允许看到当前及之前的位置
                        // 对于 prefill，位置 i 可以看到 0..i
                        if (j > i + (kvSeqLen - seqLen)) {
                            maskData[i * kvSeqLen + j] = -Infinity;
                        }
                    }
                }
                const causalMask = new Tensor(maskData, {
                    dtype: "float32",
                    shape: [1, 1, seqLen, kvSeqLen],
                });
                maskedScores = add(scaledScores, causalMask);
                scaledScores.dispose();
                causalMask.dispose();
            }

            // Step 3: Softmax
            const attnProbs = softmax(maskedScores, -1);
            if (maskedScores !== scaledScores) {
                maskedScores.dispose();
            }

            // 保存 attention weights
            // 释放之前的 weights (如果有)
            if (this.lastAttentionWeights !== null) {
                this.lastAttentionWeights.dispose();
            }
            // 克隆以保持独立生命周期
            this.lastAttentionWeights = attnProbs.clone();

            // Step 4: Attention output
            attnOutput = matmul(attnProbs, expandedV);
            attnProbs.dispose();
        } else {
            // 使用 FlashAttention (默认高性能路径)
            attnOutput = functional.scaledDotProductAttention(
                q,
                expandedK,
                expandedV,
                attnMask,
                0.0,
                effectiveIsCausal
            );

            // 清除之前的 attention weights
            if (this.lastAttentionWeights !== null) {
                this.lastAttentionWeights.dispose();
                this.lastAttentionWeights = null;
            }
        }

        // 重塑回: (batch, numHeads, seq, headDim) -> (batch, seq, numHeads * headDim)
        const reshaped = this._reshapeFromMultihead(attnOutput, bsz, seqLen);

        // 输出投影
        const output = (await this.o_proj.call(reshaped)) as Tensor;

        return output;
    }

    /**
     * 重塑张量用于多头注意力 (仅 view，不做 permute)
     * (batch, seq, numHeads * headDim) -> (batch, seq, numHeads, headDim)
     *
     * 注意：permute 需要在 Q/K Norm 之后单独调用
     */
    private _reshapeForMultihead(
        tensor: Tensor,
        bsz: number,
        seqLen: number,
        numHeads: number
    ): Tensor {
        // (batch, seq, numHeads * headDim) -> (batch, seq, numHeads, headDim)
        return view(tensor, [bsz, seqLen, numHeads, this.headDim]);
    }

    /**
     * Permute 从 BSHD 到 BHSD 并确保连续
     * (batch, seq, numHeads, headDim) -> (batch, numHeads, seq, headDim)
     */
    private _permuteToHeadsFirst(tensor: Tensor): Tensor {
        // (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        const permuted = tensor.permute([0, 2, 1, 3]);
        // 确保 contiguous 以便后续计算
        const result = contiguous(permuted);
        // ⚠️ 关键：dispose permute 产生的 view
        permuted.dispose();
        return result;
    }

    /**
     * 从多头格式恢复
     * (batch, numHeads, seq, headDim) -> (batch, seq, numHeads * headDim)
     */
    private _reshapeFromMultihead(tensor: Tensor, bsz: number, seqLen: number): Tensor {
        // (batch, numHeads, seq, headDim) -> (batch, seq, numHeads, headDim)
        const permuted = tensor.permute([0, 2, 1, 3]);

        // 确保 contiguous 后再 view
        const contiguousTensor = contiguous(permuted);
        // ⚠️ dispose permute view
        permuted.dispose();

        // (batch, seq, numHeads, headDim) -> (batch, seq, numHeads * headDim)
        const result = view(contiguousTensor, [bsz, seqLen, this.numHeads * this.headDim]);
        // contiguous 和 view 共享 storage，不需要额外 dispose

        return result;
    }

    /**
     * 扩展 KV 头以匹配 Q 头数量 (GQA 核心)
     *
     * 使用 expand + reshape 实现零拷贝扩展：
     * (batch, numKvHeads, seq, headDim)
     *   -> (batch, numKvHeads, 1, seq, headDim)           [unsqueeze]
     *   -> (batch, numKvHeads, numKvGroups, seq, headDim) [expand]
     *   -> (batch, numHeads, seq, headDim)                [reshape]
     *
     * @param x KV 张量，形状 (batch, numKvHeads, seq, headDim)
     * @returns 扩展后的张量，形状 (batch, numHeads, seq, headDim)
     */
    private _repeatKv(x: Tensor, bsz: number, seqLen: number): Tensor {
        // (batch, numKvHeads, seq, headDim) -> (batch, numKvHeads, 1, seq, headDim)
        const unsqueezed = x.unsqueeze(2);

        // (batch, numKvHeads, 1, seq, headDim) -> (batch, numKvHeads, numKvGroups, seq, headDim)
        const expanded = expand(unsqueezed, [
            bsz,
            this.numKvHeads,
            this.numKvGroups,
            seqLen,
            this.headDim,
        ]);
        // ⚠️ dispose unsqueeze view
        unsqueezed.dispose();

        // 需要 contiguous 使 expand 生效为实际数据
        const contiguousTensor = contiguous(expanded);
        // ⚠️ dispose expand view
        expanded.dispose();

        // (batch, numKvHeads, numKvGroups, seq, headDim) -> (batch, numHeads, seq, headDim)
        const result = reshape(contiguousTensor, [bsz, this.numHeads, seqLen, this.headDim]);
        // reshape 和 contiguous 共享 storage，不需要额外 dispose

        return result;
    }

    /**
     * 应用旋转位置嵌入 (RoPE)
     *
     * 使用 model-utils 中的 applyRotaryPosEmbDirect 实现
     *
     * @param q Query 张量，形状 (batch, numHeads, seq, headDim)
     * @param k Key 张量，形状 (batch, numHeads, seq, headDim)
     * @param cos Cosine 嵌入
     * @param sin Sine 嵌入
     * @returns [q_rotated, k_rotated]
     */
    private _applyRotaryEmbQK(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor): [Tensor, Tensor] {
        // 使用 rope.ts 中的实现
        return applyRotaryPosEmbDirect(q, k, cos, sin, 1);
    }

    /**
     * 额外表示信息
     */
    protected extraRepr(): string {
        const gqaInfo =
            this.numKvHeads !== this.numHeads
                ? `, numKvHeads=${this.numKvHeads}, numKvGroups=${this.numKvGroups}`
                : "";
        return `embedDim=${this.embedDim}, numHeads=${this.numHeads}, headDim=${this.headDim}${gqaInfo}`;
    }
}
