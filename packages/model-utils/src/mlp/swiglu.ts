/**
 * SwiGLU MLP - Qwen3/LLaMA3 风格的 MLP 层
 *
 * SwiGLU (Swish-Gated Linear Unit) 是现代 LLM 中常用的激活函数组合：
 * - output = down_proj(silu(gate_proj(x)) * up_proj(x))
 *
 * 对标 HuggingFace Transformers 中的 Qwen3MLP / LlamaMLP
 *
 * 主要特点：
 * - 三个独立的投影层: gate_proj, up_proj, down_proj
 * - 使用 SiLU (Swish) 作为门控激活函数
 * - 支持无偏置 (Qwen3/LLaMA3 默认不使用偏置)
 *
 * @example
 * ```ts
 * // Qwen3-0.6B: hidden=1024, intermediate=3072
 * const mlp = new SwiGLUMLP({
 *     hiddenSize: 1024,
 *     intermediateSize: 3072,
 *     bias: false,
 * });
 *
 * const output = await mlp.call(hidden_states);
 * ```
 *
 * @module @kandle/model-utils/mlp/swiglu
 */

import type { DType } from '@kandle/types';
import {
    Tensor,
    nn,
    functional,
} from '@kandle/core';

// ============================================================================
// Types
// ============================================================================

/**
 * SwiGLUMLP 构造选项
 */
export interface SwiGLUMLPOptions {
    /** 隐藏层维度 (输入/输出维度) */
    hiddenSize: number;

    /** 中间层维度 (扩展维度) */
    intermediateSize: number;

    /** 是否使用偏置 (默认 false, Qwen3/LLaMA3 不使用) */
    bias?: boolean;

    /** 数据类型 */
    dtype?: DType;
}

// ============================================================================
// SwiGLUMLP Class
// ============================================================================

/**
 * SwiGLU MLP 层
 *
 * 权重结构（适配 HuggingFace 模型）：
 * - gate_proj.weight: [intermediateSize, hiddenSize]
 * - up_proj.weight: [intermediateSize, hiddenSize]
 * - down_proj.weight: [hiddenSize, intermediateSize]
 *
 * 计算公式：
 * gate = silu(gate_proj(x))
 * up = up_proj(x)
 * hidden = gate * up  (element-wise)
 * output = down_proj(hidden)
 */
export class SwiGLUMLP extends nn.Module {
    // Config
    readonly hiddenSize: number;
    readonly intermediateSize: number;

    // 投影层
    gate_proj: nn.Linear;
    up_proj: nn.Linear;
    down_proj: nn.Linear;

    constructor(options: SwiGLUMLPOptions) {
        super();

        const {
            hiddenSize,
            intermediateSize,
            bias = false,  // Qwen3/LLaMA3 默认不使用偏置
            dtype = 'float32',
        } = options;

        this.hiddenSize = hiddenSize;
        this.intermediateSize = intermediateSize;

        // gate_proj: hiddenSize -> intermediateSize
        this.gate_proj = new nn.Linear(hiddenSize, intermediateSize, bias, dtype);

        // up_proj: hiddenSize -> intermediateSize
        this.up_proj = new nn.Linear(hiddenSize, intermediateSize, bias, dtype);

        // down_proj: intermediateSize -> hiddenSize
        this.down_proj = new nn.Linear(intermediateSize, hiddenSize, bias, dtype);

        // 注册子模块
        this.addModule('gate_proj', this.gate_proj);
        this.addModule('up_proj', this.up_proj);
        this.addModule('down_proj', this.down_proj);
    }

    /**
     * 前向传播
     *
     * @param x 输入张量，形状 (batch, seq_len, hidden_size)
     * @returns 输出张量，形状 (batch, seq_len, hidden_size)
     */
    async forward(x: Tensor): Promise<Tensor> {
        // gate = silu(gate_proj(x))
        const gateProj = await this.gate_proj.call(x) as Tensor;
        const gate = functional.silu(gateProj);

        // up = up_proj(x)
        const up = await this.up_proj.call(x) as Tensor;

        // hidden = gate * up (element-wise multiplication)
        const hidden = gate.mul(up);

        // output = down_proj(hidden)
        const output = await this.down_proj.call(hidden) as Tensor;

        return output;
    }

    /**
     * 额外表示信息
     */
    protected extraRepr(): string {
        return `hiddenSize=${this.hiddenSize}, intermediateSize=${this.intermediateSize}`;
    }
}
