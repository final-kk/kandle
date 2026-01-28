/**
 * RMSNorm - Root Mean Square 层归一化
 *
 * LLaMA 和 Qwen 等模型使用的归一化方式
 *
 * @see https://arxiv.org/abs/1910.07467
 */

import type { DType } from '@kandle/types';
import { Tensor } from '../../../tensor';
import { Module } from '../../module';
import { Parameter } from '../../parameter';
import { functional } from '../../../generated/ops';
import { ones } from '../../../generated/ops';

/**
 * RMSNorm 层构造选项
 */
export interface RMSNormOptions {
    /** 归一化形状 */
    normalizedShape: number | number[];
    /** epsilon 值，防止除零 */
    eps?: number;
    /** 数据类型 */
    dtype?: DType;
}

/**
 * RMSNorm 层
 *
 * 与 LayerNorm 类似，但不减去均值: y = x / sqrt(mean(x^2) + eps) * gamma
 *
 * @example
 * ```ts
 * const norm = new RMSNorm(4096);
 * const input = torch.randn([32, 10, 4096]);
 * const output = norm.call(input);
 * ```
 */
export class RMSNorm extends Module {
    /** 归一化形状 */
    readonly normalizedShape: number[];

    /** epsilon 值 */
    readonly eps: number;

    /** 可学习的缩放参数 */
    weight: Parameter;

    /**
     * 创建 RMSNorm 层
     */
    constructor(normalizedShape: number | number[], eps?: number, dtype?: DType);
    constructor(options: RMSNormOptions);
    constructor(
        arg0: number | number[] | RMSNormOptions,
        arg1?: number,
        arg2?: DType
    ) {
        super();

        let normalizedShape: number[];
        let eps: number = 1e-6;  // RMSNorm 通常使用更小的 eps
        let dtype: DType = 'float32';

        if (typeof arg0 === 'object' && !Array.isArray(arg0)) {
            // options 形式
            const options = arg0 as RMSNormOptions;
            normalizedShape = typeof options.normalizedShape === 'number'
                ? [options.normalizedShape]
                : options.normalizedShape;
            if (options.eps !== undefined) eps = options.eps;
            if (options.dtype !== undefined) dtype = options.dtype;
        } else {
            // 位置参数形式
            normalizedShape = typeof arg0 === 'number' ? [arg0] : arg0;
            if (arg1 !== undefined) eps = arg1;
            if (arg2 !== undefined) dtype = arg2;
        }

        this.normalizedShape = normalizedShape;
        this.eps = eps;

        // 创建缩放参数
        this.weight = new Parameter(ones(normalizedShape, dtype));
        this.registerParameter('weight', this.weight);
    }

    /**
     * 前向传播
     */
    async forward(input: Tensor): Promise<Tensor> {
        return functional.rmsNorm(
            input,
            this.normalizedShape,
            this.weight,
            this.eps
        );
    }

    /**
     * 额外表示信息
     */
    protected extraRepr(): string {
        return `${JSON.stringify(this.normalizedShape)}, eps=${this.eps}`;
    }
}
