/**
 * Tanh - 双曲正切激活函数
 *
 * f(x) = tanh(x)
 *
 * @see https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html
 */

import { Tensor } from '../../../tensor';
import { Module } from '../../module';
import { functional } from '../../../generated/ops';

/**
 * Tanh 激活函数模块
 *
 * @example
 * ```ts
 * const tanh = new Tanh();
 * const output = tanh.call(input);
 * ```
 */
export class Tanh extends Module {
    /**
     * 创建 Tanh 模块
     */
    constructor() {
        super();
    }

    /**
     * 前向传播
     */
    async forward(input: Tensor): Promise<Tensor> {
        return functional.tanh(input);
    }

    /**
     * 额外表示信息
     */
    protected extraRepr(): string {
        return '';
    }
}
