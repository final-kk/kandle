/**
 * Softmax - Softmax 激活函数
 *
 * 将输入转换为概率分布
 *
 * @see https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
 */

import { Tensor } from '../../../tensor';
import { Module } from '../../module';
import { functional } from '../../../generated/ops';

/**
 * Softmax 激活函数模块
 *
 * @example
 * ```ts
 * const softmax = new Softmax(-1); // 在最后一个维度上
 * const probs = softmax.call(logits);
 * ```
 */
export class Softmax extends Module {
    /** 应用 softmax 的维度 */
    readonly dim: number;

    /**
     * 创建 Softmax 模块
     *
     * @param dim 应用 softmax 的维度
     */
    constructor(dim: number = -1) {
        super();
        this.dim = dim;
    }

    /**
     * 前向传播
     */
    async forward(input: Tensor): Promise<Tensor> {
        return functional.softmax(input, this.dim);
    }

    /**
     * 额外表示信息
     */
    protected extraRepr(): string {
        return `dim=${this.dim}`;
    }
}
