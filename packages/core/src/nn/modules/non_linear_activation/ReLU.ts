/**
 * ReLU - 修正线性单元
 *
 * f(x) = max(0, x)
 *
 * @see https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
 */

import { Tensor } from '../../../tensor';
import { Module } from '../../module';
import { functional } from '../../../generated/ops';

/**
 * ReLU 激活函数模块
 *
 * @example
 * ```ts
 * const relu = new ReLU();
 * const output = relu.call(input);
 * ```
 */
export class ReLU extends Module {
    /**
     * 创建 ReLU 模块
     */
    constructor() {
        super();
    }

    /**
     * 前向传播
     */
    async forward(input: Tensor): Promise<Tensor> {
        return functional.relu(input);
    }

    /**
     * 额外表示信息
     */
    protected extraRepr(): string {
        return '';
    }
}
