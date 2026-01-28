/**
 * ResidualAttentionBlock - Whisper 残差注意力块
 *
 * 对标 OpenAI Whisper 中的 ResidualAttentionBlock
 *
 * 结构 (Encoder 版本，无 cross-attention)：
 * - LayerNorm → MultiHeadAttention → Residual
 * - LayerNorm → MLP (Linear → GELU → Linear) → Residual
 *
 * 结构 (Decoder 版本，有 cross-attention)：
 * - LayerNorm → Self-Attention (causal) → Residual
 * - LayerNorm → Cross-Attention → Residual
 * - LayerNorm → MLP → Residual
 *
 * @module @kandle/model-utils/whisper/block
 */

import type { DType } from '@kandle/types';
import { Tensor, nn, add } from '@kandle/core';

// ============================================================================
// Types
// ============================================================================

export interface ResidualAttentionBlockConfig {
    /** 模型维度 (n_state) */
    nState: number;

    /** 注意力头数 */
    nHead: number;

    /** 是否包含 cross-attention (Decoder 使用) */
    crossAttention?: boolean;

    /** 数据类型 */
    dtype?: DType;
}

export interface ResidualAttentionBlockForwardOptions {
    /** Cross-attention 的 key/value 来源 (encoder 输出) */
    xa?: Tensor;

    /** 因果掩码 (Decoder self-attention 使用) */
    mask?: Tensor;
}

// ============================================================================
// ResidualAttentionBlock Class
// ============================================================================

/**
 * Whisper 残差注意力块
 *
 * HuggingFace 权重结构 (与 OpenAI 不同)：
 * Encoder:
 * - self_attn.k_proj, self_attn.v_proj, self_attn.q_proj, self_attn.out_proj
 * - self_attn_layer_norm
 * - fc1, fc2
 * - final_layer_norm
 *
 * Decoder (带 cross-attention):
 * - self_attn.*, self_attn_layer_norm
 * - encoder_attn.*, encoder_attn_layer_norm
 * - fc1, fc2
 * - final_layer_norm
 */
export class ResidualAttentionBlock extends nn.Module {
    // Config
    readonly nState: number;
    readonly nHead: number;

    // Self-Attention
    self_attn: nn.MultiheadAttention;
    self_attn_layer_norm: nn.LayerNorm;

    // Cross-Attention (仅 Decoder，HF 称为 encoder_attn)
    encoder_attn: nn.MultiheadAttention | null = null;
    encoder_attn_layer_norm: nn.LayerNorm | null = null;

    // MLP (HF 使用 fc1, fc2)
    fc1: nn.Linear;
    fc2: nn.Linear;
    final_layer_norm: nn.LayerNorm;

    // 内部使用
    private readonly gelu: nn.GELU;

    constructor(config: ResidualAttentionBlockConfig) {
        super();

        const {
            nState,
            nHead,
            crossAttention = false,
            dtype = 'float32',
        } = config;

        this.nState = nState;
        this.nHead = nHead;

        // Self-Attention
        // Whisper 使用标准 MHA，key 无 bias
        this.self_attn = new nn.MultiheadAttention({
            embedDim: nState,
            numHeads: nHead,
            bias: true,
            batchFirst: true,
            dtype,
        });
        this.self_attn_layer_norm = new nn.LayerNorm([nState]);

        // Cross-Attention (Decoder 专用, HF 称为 encoder_attn)
        if (crossAttention) {
            this.encoder_attn = new nn.MultiheadAttention({
                embedDim: nState,
                numHeads: nHead,
                bias: true,
                batchFirst: true,
                dtype,
            });
            this.encoder_attn_layer_norm = new nn.LayerNorm([nState]);
        }

        // MLP: n_state → 4*n_state → n_state (HF 使用 fc1, fc2)
        const nMlp = nState * 4;
        this.fc1 = new nn.Linear(nState, nMlp, true, dtype);
        this.gelu = new nn.GELU();
        this.fc2 = new nn.Linear(nMlp, nState, true, dtype);
        this.final_layer_norm = new nn.LayerNorm([nState]);

        // 注册子模块（与 HuggingFace 权重结构对齐）
        this.addModule('self_attn', this.self_attn);
        this.addModule('self_attn_layer_norm', this.self_attn_layer_norm);
        if (this.encoder_attn) {
            this.addModule('encoder_attn', this.encoder_attn);
            this.addModule('encoder_attn_layer_norm', this.encoder_attn_layer_norm);
        }
        this.addModule('fc1', this.fc1);
        this.addModule('fc2', this.fc2);
        this.addModule('final_layer_norm', this.final_layer_norm);
    }

    /**
     * 前向传播
     *
     * @param x - 输入张量，形状 (batch, seq_len, n_state)
     * @param options - 可选参数
     * @returns 输出张量，形状 (batch, seq_len, n_state)
     */
    async forward(
        x: Tensor,
        options?: ResidualAttentionBlockForwardOptions
    ): Promise<Tensor> {
        const { xa, mask } = options ?? {};

        // ==========================================
        // Self-Attention Block
        // ==========================================
        // x = x + self_attn(self_attn_layer_norm(x))
        const normed1 = await this.self_attn_layer_norm.call(x) as Tensor;

        const { attnOutput: attn1 } = await this.self_attn.call(
            normed1,
            normed1,
            normed1,
            undefined,  // keyPaddingMask
            false,      // needWeights
            mask,       // attnMask
        ) as { attnOutput: Tensor; attnWeights: Tensor | null };

        x = add(x, attn1);

        // ==========================================
        // Cross-Attention Block (Decoder only)
        // ==========================================
        if (this.encoder_attn && this.encoder_attn_layer_norm && xa) {
            // x = x + encoder_attn(encoder_attn_layer_norm(x), xa)
            const normed2 = await this.encoder_attn_layer_norm.call(x) as Tensor;
            const { attnOutput: attn2 } = await this.encoder_attn.call(
                normed2,  // query
                xa,       // key
                xa,       // value
                undefined,
                false,
            ) as { attnOutput: Tensor; attnWeights: Tensor | null };
            x = add(x, attn2);
        }

        // ==========================================
        // MLP Block
        // ==========================================
        // x = x + mlp(final_layer_norm(x))
        const normed3 = await this.final_layer_norm.call(x) as Tensor;
        let mlpOut = await this.fc1.call(normed3) as Tensor;
        mlpOut = await this.gelu.call(mlpOut) as Tensor;
        mlpOut = await this.fc2.call(mlpOut) as Tensor;
        x = add(x, mlpOut);

        return x;
    }
}
