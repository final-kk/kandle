/**
 * MultiheadAttentionFast - 便捷推理版本的多头注意力模块
 *
 * 简化接口，专为推理场景优化：
 * - 使用分开的 Q/K/V 投影（更适合加载 HuggingFace 模型权重）
 * - 默认 batch_first=true（更符合现代使用习惯）
 * - 简化的 forward 签名
 * - 移除训练相关参数 (dropout)
 *
 * @example
 * ```ts
 * const attn = new MultiheadAttentionFast({ embedDim: 512, numHeads: 8 });
 * const output = attn.call(query, key, value);
 * // 或使用因果掩码
 * const output = attn.call(query, key, value, { isCausal: true });
 * ```
 */

import type { DType } from '@kandle/types';
import { Tensor } from '../../../tensor';
import { Module } from '../../module';
import { Linear } from '../linear';
import { functional } from '../../../generated/ops';
import { view, transpose, contiguous } from '../../../generated/ops';

// ============================================================================
// Types
// ============================================================================

/**
 * MultiheadAttentionFast 构造选项
 */
export interface MultiheadAttentionFastOptions {
    /** 模型总维度 (必需) */
    embedDim: number;

    /** 注意力头数量 (必需) */
    numHeads: number;

    /** 是否为投影层添加偏置 (默认 true) */
    bias?: boolean;

    /** Key/Value 的维度，如果与 embedDim 不同 */
    kvDim?: number;

    /** 数据类型 */
    dtype?: DType;
}

/**
 * Forward 选项
 */
export interface MultiheadAttentionFastForwardOptions {
    /** 注意力掩码 (additive, float) */
    attnMask?: Tensor;

    /** 是否使用因果掩码 */
    isCausal?: boolean;
}

// ============================================================================
// MultiheadAttentionFast Class
// ============================================================================

/**
 * MultiheadAttentionFast - 便捷推理版本的多头注意力
 *
 * 与 PyTorch MHA 的主要区别：
 * - 使用分开的 Linear 子模块 (q_proj, k_proj, v_proj, out_proj)
 * - 默认假设输入格式为 (batch, seq, embed_dim)
 * - 简化的 forward 签名
 *
 * 权重结构（便于加载 HuggingFace 模型）：
 * - q_proj.weight: [embed_dim, embed_dim]
 * - q_proj.bias: [embed_dim]
 * - k_proj.weight: [embed_dim, kv_dim]
 * - k_proj.bias: [embed_dim]
 * - v_proj.weight: [embed_dim, kv_dim]
 * - v_proj.bias: [embed_dim]
 * - out_proj.weight: [embed_dim, embed_dim]
 * - out_proj.bias: [embed_dim]
 */
export class MultiheadAttentionFast extends Module {
    // Config
    readonly embedDim: number;
    readonly numHeads: number;
    readonly headDim: number;
    readonly kvDim: number;

    // 投影层 (作为子模块，便于 state_dict 加载)
    q_proj: Linear;
    k_proj: Linear;
    v_proj: Linear;
    out_proj: Linear;

    constructor(options: MultiheadAttentionFastOptions);
    constructor(embedDim: number, numHeads: number, options?: Partial<MultiheadAttentionFastOptions>);
    constructor(
        arg0: number | MultiheadAttentionFastOptions,
        arg1?: number | Partial<MultiheadAttentionFastOptions>,
        arg2?: Partial<MultiheadAttentionFastOptions>
    ) {
        super();

        let embedDim: number;
        let numHeads: number;
        let bias = true;
        let kvDim: number | undefined;
        let dtype: DType = 'float32';

        // 解析参数
        if (typeof arg0 === 'object') {
            embedDim = arg0.embedDim;
            numHeads = arg0.numHeads;
            if (arg0.bias !== undefined) bias = arg0.bias;
            kvDim = arg0.kvDim;
            if (arg0.dtype !== undefined) dtype = arg0.dtype;
        } else {
            embedDim = arg0;
            numHeads = arg1 as number;
            const opts = arg2 ?? {};
            if (opts.bias !== undefined) bias = opts.bias;
            kvDim = opts.kvDim;
            if (opts.dtype !== undefined) dtype = opts.dtype;
        }

        // 验证参数
        if (embedDim % numHeads !== 0) {
            throw new Error(
                `embed_dim must be divisible by num_heads, got embed_dim=${embedDim}, num_heads=${numHeads}`
            );
        }

        this.embedDim = embedDim;
        this.numHeads = numHeads;
        this.headDim = Math.floor(embedDim / numHeads);
        this.kvDim = kvDim ?? embedDim;

        // 创建投影层
        this.q_proj = new Linear(embedDim, embedDim, bias, dtype);
        this.k_proj = new Linear(this.kvDim, embedDim, bias, dtype);
        this.v_proj = new Linear(this.kvDim, embedDim, bias, dtype);
        this.out_proj = new Linear(embedDim, embedDim, bias, dtype);

        // 注册子模块
        this.addModule('q_proj', this.q_proj);
        this.addModule('k_proj', this.k_proj);
        this.addModule('v_proj', this.v_proj);
        this.addModule('out_proj', this.out_proj);
    }

    /**
     * 前向传播
     *
     * @param query Query 张量，形状 (batch, tgt_len, embed_dim)
     * @param key Key 张量，形状 (batch, src_len, kv_dim)
     * @param value Value 张量，形状 (batch, src_len, kv_dim)
     * @param options 可选参数：attnMask, isCausal
     * @returns 输出张量，形状 (batch, tgt_len, embed_dim)
     */
    async forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        options?: MultiheadAttentionFastForwardOptions
    ): Promise<Tensor> {
        const { attnMask, isCausal = false } = options ?? {};

        // 检查 is_causal 与 attn_mask 互斥
        if (isCausal && attnMask !== undefined) {
            throw new Error('isCausal=true cannot be used with attnMask');
        }

        // 获取维度
        const bsz = query.shape[0];
        const tgtLen = query.shape[1];
        const srcLen = key.shape[1];

        // 线性投影 (await 异步调用)
        // (batch, seq, dim) -> (batch, seq, embed_dim)
        let qProj = await this.q_proj.call(query) as Tensor;
        let kProj = await this.k_proj.call(key) as Tensor;
        let vProj = await this.v_proj.call(value) as Tensor;

        // 重塑为多头
        // (batch, seq, embed_dim) -> (batch, num_heads, seq, head_dim)
        qProj = this._reshapeForMultihead(qProj, bsz, tgtLen);
        kProj = this._reshapeForMultihead(kProj, bsz, srcLen);
        vProj = this._reshapeForMultihead(vProj, bsz, srcLen);

        // Scaled Dot-Product Attention
        // (batch, num_heads, L, head_dim) @ (batch, num_heads, head_dim, S) -> (batch, num_heads, L, S)
        const attnOutput = functional.scaledDotProductAttention(
            qProj,
            kProj,
            vProj,
            attnMask,
            0.0, // dropout
            isCausal
        );

        // 重塑回: (batch, num_heads, tgt_len, head_dim) -> (batch, tgt_len, embed_dim)
        let output = this._reshapeFromMultihead(attnOutput, bsz, tgtLen);

        // 输出投影 (await 异步调用)
        output = await this.out_proj.call(output) as Tensor;

        return output;
    }

    /**
     * 重塑张量用于多头注意力
     * (batch, seq, embed_dim) -> (batch, num_heads, seq, head_dim)
     */
    private _reshapeForMultihead(tensor: Tensor, bsz: number, seqLen: number): Tensor {
        // (batch, seq, embed_dim) -> (batch, seq, num_heads, head_dim)
        let reshaped = view(tensor, [bsz, seqLen, this.numHeads, this.headDim]);

        // (batch, seq, num_heads, head_dim) -> (batch, num_heads, seq, head_dim)
        reshaped = reshaped.permute([0, 2, 1, 3]);

        // 确保 contiguous 以便后续计算
        reshaped = contiguous(reshaped);

        return reshaped;
    }

    /**
     * 从多头格式恢复
     * (batch, num_heads, seq, head_dim) -> (batch, seq, embed_dim)
     */
    private _reshapeFromMultihead(tensor: Tensor, bsz: number, seqLen: number): Tensor {
        // (batch, num_heads, seq, head_dim) -> (batch, seq, num_heads, head_dim)
        let reshaped = tensor.permute([0, 2, 1, 3]);

        // 确保 contiguous 后再 view
        reshaped = contiguous(reshaped);

        // (batch, seq, num_heads, head_dim) -> (batch, seq, embed_dim)
        reshaped = view(reshaped, [bsz, seqLen, this.embedDim]);

        return reshaped;
    }

    /**
     * 额外表示信息
     */
    protected extraRepr(): string {
        return `embed_dim=${this.embedDim}, num_heads=${this.numHeads}, kv_dim=${this.kvDim}`;
    }
}
