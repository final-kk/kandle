/**
 * SiLU - Sigmoid Linear Unit (Swish)
 *
 * f(x) = x * sigmoid(x)
 *
 * LLaMA 的 SwiGLU 中使用的激活函数
 *
 * @see https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
 */

import { Tensor } from '../../../tensor';
import { Module } from '../../module';
import { functional } from '../../../generated/ops';

/**
 * SiLU 激活函数模块
 *
 * 也称为 Swish 激活函数
 *
 * @example
 * ```ts
 * const silu = new SiLU();
 * const output = silu.call(input);
 * ```
 */
export class SiLU extends Module {
    /**
     * 创建 SiLU 模块
     */
    constructor() {
        super();
    }

    /**
     * 前向传播
     */
    async forward(input: Tensor): Promise<Tensor> {
        return functional.silu(input);
    }

    /**
     * 额外表示信息
     */
    protected extraRepr(): string {
        return '';
    }
}
