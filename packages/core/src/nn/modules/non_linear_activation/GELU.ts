/**
 * GELU - 高斯误差线性单元
 *
 * Whisper 等模型使用的激活函数
 *
 * @see https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
 */

import { Tensor } from '../../../tensor';
import { Module } from '../../module';
import { functional } from '../../../generated/ops';

/**
 * GELU 激活函数模块
 *
 * @example
 * ```ts
 * const gelu = new GELU();
 * const output = gelu.call(input);
 * ```
 */
export class GELU extends Module {
    /** 近似方法: 'none' | 'tanh' */
    readonly approximate: 'none' | 'tanh';

    /**
     * 创建 GELU 模块
     *
     * @param approximate 近似方法，默认 'none' (精确计算)
     */
    constructor(approximate: 'none' | 'tanh' = 'none') {
        super();
        this.approximate = approximate;
    }

    /**
     * 前向传播
     */
    async forward(input: Tensor): Promise<Tensor> {
        return functional.gelu(input, this.approximate);
    }

    /**
     * 额外表示信息
     */
    protected extraRepr(): string {
        return `approximate='${this.approximate}'`;
    }
}
