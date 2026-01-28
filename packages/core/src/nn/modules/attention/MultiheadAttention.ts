/**
 * MultiheadAttention - 多头注意力模块
 *
 * 严格对标 PyTorch torch.nn.MultiheadAttention
 *
 * 支持自注意力和交叉注意力，通过线性投影将 Q, K, V 映射到多个头，
 * 对每个头应用 Scaled Dot-Product Attention，然后拼接并通过输出投影。
 *
 * @see https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
 */

import type { DType } from '@kandle/types';
import { Tensor } from '../../../tensor';
import { Module } from '../../module';
import { Parameter } from '../../parameter';
import { Linear } from '../linear';
import { functional } from '../../../generated/ops';
import { empty, zeros, slice, view, transpose, contiguous, reshape } from '../../../generated/ops';

// ============================================================================
// Types
// ============================================================================

/**
 * MultiheadAttention 构造选项
 */
export interface MultiheadAttentionOptions {
    /** 模型总维度 (必需) */
    embedDim: number;

    /** 注意力头数量 (必需) */
    numHeads: number;

    /** Dropout 概率 (默认 0.0，当前推理模式下忽略) */
    dropout?: number;

    /** 是否为投影层添加偏置 (默认 true) */
    bias?: boolean;

    /** 是否为 K/V 添加偏置 (默认 false，当前不支持) */
    addBiasKv?: boolean;

    /** 是否为 K/V 添加零填充 (默认 false，当前不支持) */
    addZeroAttn?: boolean;

    /** Key 的维度，如果与 embedDim 不同 */
    kdim?: number;

    /** Value 的维度，如果与 embedDim 不同 */
    vdim?: number;

    /** 输入张量格式是否为 (batch, seq, feature) (默认 false) */
    batchFirst?: boolean;

    /** 数据类型 */
    dtype?: DType;
}

/**
 * MultiheadAttention forward 结果
 */
export interface MultiheadAttentionOutput {
    /** 注意力输出 */
    attnOutput: Tensor;

    /** 注意力权重 (如果 need_weights=true) */
    attnWeights: Tensor | null;
}

// ============================================================================
// MultiheadAttention Class
// ============================================================================

/**
 * MultiheadAttention - 多头注意力模块
 *
 * 对标 PyTorch nn.MultiheadAttention，使用合并的 in_proj_weight 方式
 *
 * 权重结构:
 * - in_proj_weight: [3 * embed_dim, embed_dim] (当 kdim=vdim=embed_dim)
 * - in_proj_bias: [3 * embed_dim]
 * - out_proj.weight: [embed_dim, embed_dim]
 * - out_proj.bias: [embed_dim]
 *
 * 或者当 kdim/vdim 不同时:
 * - q_proj_weight: [embed_dim, embed_dim]
 * - k_proj_weight: [embed_dim, kdim]
 * - v_proj_weight: [embed_dim, vdim]
 *
 * @example
 * ```ts
 * const mha = new MultiheadAttention({ embedDim: 512, numHeads: 8 });
 * const { attnOutput, attnWeights } = mha.call(query, key, value);
 * ```
 */
export class MultiheadAttention extends Module {
    // Config
    readonly embedDim: number;
    readonly numHeads: number;
    readonly headDim: number;
    readonly dropout: number;
    readonly batchFirst: boolean;
    readonly kdim: number;
    readonly vdim: number;

    // 是否使用合并投影 (kdim === vdim === embedDim)
    private readonly _qkvSameEmbedDim: boolean;

    // 合并投影参数 (当 _qkvSameEmbedDim=true)
    in_proj_weight: Parameter | null = null;
    in_proj_bias: Parameter | null = null;

    // 分开投影参数 (当 _qkvSameEmbedDim=false)
    q_proj_weight: Parameter | null = null;
    k_proj_weight: Parameter | null = null;
    v_proj_weight: Parameter | null = null;

    // 偏置 (分开投影时)
    private _useBias: boolean;

    // 输出投影 (作为子模块)
    out_proj: Linear;

    // 偏置 K/V (当前不支持)
    bias_k: Parameter | null = null;
    bias_v: Parameter | null = null;

    constructor(options: MultiheadAttentionOptions);
    constructor(embedDim: number, numHeads: number, options?: Partial<MultiheadAttentionOptions>);
    constructor(
        arg0: number | MultiheadAttentionOptions,
        arg1?: number | Partial<MultiheadAttentionOptions>,
        arg2?: Partial<MultiheadAttentionOptions>
    ) {
        super();

        let embedDim: number;
        let numHeads: number;
        let dropout = 0.0;
        let bias = true;
        let addBiasKv = false;
        let addZeroAttn = false;
        let kdim: number | undefined;
        let vdim: number | undefined;
        let batchFirst = false;
        let dtype: DType = 'float32';

        // 解析参数
        if (typeof arg0 === 'object') {
            // options 形式
            embedDim = arg0.embedDim;
            numHeads = arg0.numHeads;
            if (arg0.dropout !== undefined) dropout = arg0.dropout;
            if (arg0.bias !== undefined) bias = arg0.bias;
            if (arg0.addBiasKv !== undefined) addBiasKv = arg0.addBiasKv;
            if (arg0.addZeroAttn !== undefined) addZeroAttn = arg0.addZeroAttn;
            kdim = arg0.kdim;
            vdim = arg0.vdim;
            if (arg0.batchFirst !== undefined) batchFirst = arg0.batchFirst;
            if (arg0.dtype !== undefined) dtype = arg0.dtype;
        } else {
            // 位置参数形式
            embedDim = arg0;
            numHeads = arg1 as number;
            const opts = arg2 ?? {};
            if (opts.dropout !== undefined) dropout = opts.dropout;
            if (opts.bias !== undefined) bias = opts.bias;
            if (opts.addBiasKv !== undefined) addBiasKv = opts.addBiasKv;
            if (opts.addZeroAttn !== undefined) addZeroAttn = opts.addZeroAttn;
            kdim = opts.kdim;
            vdim = opts.vdim;
            if (opts.batchFirst !== undefined) batchFirst = opts.batchFirst;
            if (opts.dtype !== undefined) dtype = opts.dtype;
        }

        // 验证参数
        if (embedDim % numHeads !== 0) {
            throw new Error(
                `embed_dim must be divisible by num_heads, got embed_dim=${embedDim}, num_heads=${numHeads}`
            );
        }

        // 不支持的功能警告
        if (addBiasKv) {
            console.warn('MultiheadAttention: add_bias_kv is not supported yet');
        }
        if (addZeroAttn) {
            console.warn('MultiheadAttention: add_zero_attn is not supported yet');
        }

        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = Math.floor(embedDim / numHeads);
        this.dropout = dropout;
        this.batchFirst = batchFirst;
        this.kdim = kdim ?? embedDim;
        this.vdim = vdim ?? embedDim;
        this._useBias = bias;

        this._qkvSameEmbedDim = this.kdim === embedDim && this.vdim === embedDim;

        // 创建参数
        if (this._qkvSameEmbedDim) {
            // 合并投影: in_proj_weight [3*embed_dim, embed_dim]
            this.in_proj_weight = new Parameter(
                empty([3 * embedDim, embedDim], dtype)
            );
            this.registerParameter('in_proj_weight', this.in_proj_weight);

            if (bias) {
                this.in_proj_bias = new Parameter(
                    zeros([3 * embedDim], dtype)
                );
                this.registerParameter('in_proj_bias', this.in_proj_bias);
            } else {
                this.registerParameter('in_proj_bias', null);
            }

            // 不需要分开的参数
            this.registerParameter('q_proj_weight', null);
            this.registerParameter('k_proj_weight', null);
            this.registerParameter('v_proj_weight', null);
        } else {
            // 分开投影
            this.q_proj_weight = new Parameter(
                empty([embedDim, embedDim], dtype)
            );
            this.k_proj_weight = new Parameter(
                empty([embedDim, this.kdim], dtype)
            );
            this.v_proj_weight = new Parameter(
                empty([embedDim, this.vdim], dtype)
            );
            this.registerParameter('q_proj_weight', this.q_proj_weight);
            this.registerParameter('k_proj_weight', this.k_proj_weight);
            this.registerParameter('v_proj_weight', this.v_proj_weight);

            // 分开投影模式下没有 in_proj
            this.registerParameter('in_proj_weight', null);
            this.registerParameter('in_proj_bias', null);
        }

        // 输出投影 (作为子模块)
        this.out_proj = new Linear(embedDim, embedDim, bias, dtype);
        this.addModule('out_proj', this.out_proj);

        // 初始化权重
        this._resetParameters();
    }

    /**
     * 重置参数 (初始化)
     *
     * PyTorch 使用 xavier_uniform_ 初始化
     * 当前简化实现，等待实际加载预训练权重
     */
    private _resetParameters(): void {
        // TODO: 实现 xavier_uniform_ 初始化
        // 当前权重保持未初始化状态
        // 加载预训练权重时会被覆盖
    }

    /**
     * 前向传播
     *
     * @param query Query 张量，形状取决于 batch_first:
     *              - batch_first=False: (L, N, E) 或 (L, E) (unbatched)
     *              - batch_first=True: (N, L, E)
     * @param key Key 张量，形状:
     *              - batch_first=False: (S, N, E) 或 (S, E)
     *              - batch_first=True: (N, S, E)
     * @param value Value 张量，形状与 key 相同
     * @param keyPaddingMask (可选) Key 的 padding mask，形状 (N, S)
     * @param needWeights 是否返回注意力权重 (默认 true)
     * @param attnMask (可选) 注意力掩码，形状 (L, S) 或 (N*numHeads, L, S)
     * @param averageAttnWeights 是否平均注意力权重 (默认 true)
     * @param isCausal 是否使用因果掩码 (默认 false)
     * @returns { attnOutput, attnWeights }
     */
    async forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        keyPaddingMask?: Tensor,
        needWeights: boolean = true,
        attnMask?: Tensor,
        averageAttnWeights: boolean = true,
        isCausal: boolean = false
    ): Promise<MultiheadAttentionOutput> {
        // 检查 is_causal 与 attn_mask 不能同时使用
        if (isCausal && attnMask !== undefined) {
            throw new Error(
                'MultiheadAttention: is_causal=true cannot be used with attn_mask'
            );
        }

        // 处理 batch_first
        let q = query;
        let k = key;
        let v = value;

        if (this.batchFirst) {
            // (N, L, E) -> (L, N, E)
            q = transpose(q, 0, 1);
            k = transpose(k, 0, 1);
            v = transpose(v, 0, 1);
        }

        // 获取维度
        const is3D = q.shape.length === 3;
        const tgtLen = q.shape[0];
        const bsz = is3D ? q.shape[1] : 1;
        const srcLen = k.shape[0];

        // 如果是 2D (unbatched), 需要扩展
        if (!is3D) {
            // (L, E) -> (L, 1, E)
            q = view(q, [tgtLen, 1, this.embedDim]);
            k = view(k, [srcLen, 1, this.kdim]);
            v = view(v, [srcLen, 1, this.vdim]);
        }

        // 线性投影
        let qProj: Tensor, kProj: Tensor, vProj: Tensor;

        if (this._qkvSameEmbedDim) {
            // 使用合并投影 - 分别提取权重部分
            // in_proj_weight: [3*E, E] -> 分成 W_q [E, E], W_k [E, E], W_v [E, E]
            const E = this.embedDim;

            // 通过 slice 提取权重
            // Python 语法: in_proj_weight[0:E, :]
            const qWeight = slice(this.in_proj_weight!, `0:${E}, :`);
            const kWeight = slice(this.in_proj_weight!, `${E}:${2 * E}, :`);
            const vWeight = slice(this.in_proj_weight!, `${2 * E}:${3 * E}, :`);

            let qBias: Tensor | undefined, kBias: Tensor | undefined, vBias: Tensor | undefined;
            if (this.in_proj_bias) {
                // in_proj_bias: [3*E] -> 分成 b_q [E], b_k [E], b_v [E]
                qBias = slice(this.in_proj_bias!, `0:${E}`);
                kBias = slice(this.in_proj_bias!, `${E}:${2 * E}`);
                vBias = slice(this.in_proj_bias!, `${2 * E}:${3 * E}`);
            }

            // 投影: (L, N, E) @ (E, E).T + b -> (L, N, E)
            qProj = functional.linear(q, qWeight, qBias);
            kProj = functional.linear(k, kWeight, kBias);
            vProj = functional.linear(v, vWeight, vBias);
        } else {
            // 使用分开投影
            qProj = functional.linear(q, this.q_proj_weight!);
            kProj = functional.linear(k, this.k_proj_weight!);
            vProj = functional.linear(v, this.v_proj_weight!);
        }

        // 重塑为多头: (L, N, E) -> (N*numHeads, L, headDim)
        qProj = this._reshapeForMultihead(qProj, bsz, tgtLen);
        kProj = this._reshapeForMultihead(kProj, bsz, srcLen);
        vProj = this._reshapeForMultihead(vProj, bsz, srcLen);

        // 处理 key_padding_mask
        let combinedMask = attnMask;
        if (keyPaddingMask !== undefined) {
            // 将 key_padding_mask (N, S) 转换为 attention mask 格式
            // TODO: 实现 key_padding_mask 处理
            console.warn('MultiheadAttention: key_padding_mask is not fully supported yet');
        }

        // 调用 scaled_dot_product_attention
        // qProj, kProj, vProj 形状: (N*numHeads, L or S, headDim)
        const attnOutput = functional.scaledDotProductAttention(
            qProj,
            kProj,
            vProj,
            combinedMask,
            0.0, // dropout (ignored in inference)
            isCausal
        );

        // 重塑回: (N*numHeads, L, headDim) -> (L, N, E)
        let output = this._reshapeFromMultihead(attnOutput, bsz, tgtLen);

        // 输出投影 (await 异步调用)
        output = await this.out_proj.call(output) as Tensor;

        // 处理 batch_first
        if (this.batchFirst && is3D) {
            output = transpose(output, 0, 1);
        }

        // 处理 unbatched 情况
        if (!is3D) {
            output = view(output, [tgtLen, this.embedDim]);
        }

        // 注意力权重 (当前简化实现，不返回实际权重)
        const attnWeights = needWeights ? null : null; // TODO: 实现 attention weights 返回

        return { attnOutput: output, attnWeights };
    }

    /**
     * 重塑张量用于多头注意力
     * (L, N, E) -> (N*numHeads, L, headDim)
     */
    private _reshapeForMultihead(tensor: Tensor, bsz: number, seqLen: number): Tensor {
        // (L, N, E) -> (L, N, numHeads, headDim)
        let reshaped = view(tensor, [seqLen, bsz, this.numHeads, this.headDim]);

        // (L, N, numHeads, headDim) -> (N, numHeads, L, headDim)
        reshaped = reshaped.permute([1, 2, 0, 3]);

        // (N, numHeads, L, headDim) -> (N*numHeads, L, headDim)
        // 需要确保 contiguous 后再 view
        reshaped = contiguous(reshaped);
        reshaped = view(reshaped, [bsz * this.numHeads, seqLen, this.headDim]);

        return reshaped;
    }

    /**
     * 从多头格式恢复
     * (N*numHeads, L, headDim) -> (L, N, E)
     */
    private _reshapeFromMultihead(tensor: Tensor, bsz: number, seqLen: number): Tensor {
        // (N*numHeads, L, headDim) -> (N, numHeads, L, headDim)
        let reshaped = view(tensor, [bsz, this.numHeads, seqLen, this.headDim]);

        // (N, numHeads, L, headDim) -> (L, N, numHeads, headDim)
        reshaped = reshaped.permute([2, 0, 1, 3]);

        // (L, N, numHeads, headDim) -> (L, N, E)
        reshaped = contiguous(reshaped);
        reshaped = view(reshaped, [seqLen, bsz, this.embedDim]);

        return reshaped;
    }

    /**
     * 额外表示信息
     */
    protected extraRepr(): string {
        return (
            `embed_dim=${this.embedDim}, num_heads=${this.numHeads}, ` +
            `dropout=${this.dropout}, batch_first=${this.batchFirst}`
        );
    }
}
