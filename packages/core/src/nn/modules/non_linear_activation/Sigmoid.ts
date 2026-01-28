/**
 * Sigmoid - Sigmoid 激活函数
 *
 * f(x) = 1 / (1 + exp(-x))
 *
 * @see https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html
 */

import { Tensor } from '../../../tensor';
import { Module } from '../../module';
import { functional } from '../../../generated/ops';

/**
 * Sigmoid 激活函数模块
 *
 * @example
 * ```ts
 * const sigmoid = new Sigmoid();
 * const output = sigmoid.call(input);
 * ```
 */
export class Sigmoid extends Module {
    /**
     * 创建 Sigmoid 模块
     */
    constructor() {
        super();
    }

    /**
     * 前向传播
     */
    async forward(input: Tensor): Promise<Tensor> {
        return functional.sigmoid(input);
    }

    /**
     * 额外表示信息
     */
    protected extraRepr(): string {
        return '';
    }
}
