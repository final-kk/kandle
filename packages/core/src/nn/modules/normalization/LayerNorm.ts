/**
 * LayerNorm - 层归一化
 *
 * 对最后几个维度进行归一化
 *
 * @see https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
 */

import type { DType, Shape } from '@kandle/types';
import { Tensor } from '../../../tensor';
import { Module } from '../../module';
import { Parameter } from '../../parameter';
import { functional } from '../../../generated/ops';
import { ones, zeros } from '../../../generated/ops';

/**
 * LayerNorm 层构造选项
 */
export interface LayerNormOptions {
    /** 归一化形状，可以是单个数字或形状数组 */
    normalizedShape: number | number[];
    /** epsilon 值，防止除零 */
    eps?: number;
    /** 是否使用可学习的仿射参数 */
    elementwiseAffine?: boolean;
    /** 数据类型 */
    dtype?: DType;
}

/**
 * LayerNorm 层
 *
 * 对最后几个维度进行归一化: y = (x - mean) / sqrt(var + eps) * gamma + beta
 *
 * @example
 * ```ts
 * // 对最后一个维度 (256) 进行归一化
 * const norm = new LayerNorm(256);
 * const input = torch.randn([32, 10, 256]);
 * const output = norm.call(input);
 * ```
 */
export class LayerNorm extends Module {
    /** 归一化形状 */
    readonly normalizedShape: number[];

    /** epsilon 值 */
    readonly eps: number;

    /** 是否使用仿射参数 */
    readonly elementwiseAffine: boolean;

    /** 可学习的缩放参数 gamma */
    weight: Parameter | null;

    /** 可学习的偏移参数 beta */
    bias: Parameter | null;

    /**
     * 创建 LayerNorm 层
     */
    constructor(normalizedShape: number | number[], eps?: number, elementwiseAffine?: boolean, dtype?: DType);
    constructor(options: LayerNormOptions);
    constructor(
        arg0: number | number[] | LayerNormOptions,
        arg1?: number,
        arg2?: boolean,
        arg3?: DType
    ) {
        super();

        let normalizedShape: number[];
        let eps: number = 1e-5;
        let elementwiseAffine: boolean = true;
        let dtype: DType = 'float32';

        if (typeof arg0 === 'object' && !Array.isArray(arg0)) {
            // options 形式
            const options = arg0 as LayerNormOptions;
            normalizedShape = typeof options.normalizedShape === 'number'
                ? [options.normalizedShape]
                : options.normalizedShape;
            if (options.eps !== undefined) eps = options.eps;
            if (options.elementwiseAffine !== undefined) elementwiseAffine = options.elementwiseAffine;
            if (options.dtype !== undefined) dtype = options.dtype;
        } else {
            // 位置参数形式
            normalizedShape = typeof arg0 === 'number' ? [arg0] : arg0;
            if (arg1 !== undefined) eps = arg1;
            if (arg2 !== undefined) elementwiseAffine = arg2;
            if (arg3 !== undefined) dtype = arg3;
        }

        this.normalizedShape = normalizedShape;
        this.eps = eps;
        this.elementwiseAffine = elementwiseAffine;

        if (elementwiseAffine) {
            // 创建可学习参数
            this.weight = new Parameter(ones(normalizedShape, dtype));
            this.bias = new Parameter(zeros(normalizedShape, dtype));
            this.registerParameter('weight', this.weight);
            this.registerParameter('bias', this.bias);
        } else {
            this.weight = null;
            this.bias = null;
            this.registerParameter('weight', null);
            this.registerParameter('bias', null);
        }
    }

    /**
     * 前向传播
     */
    async forward(input: Tensor): Promise<Tensor> {
        return functional.layerNorm(
            input,
            this.normalizedShape,
            this.weight ?? undefined,
            this.bias ?? undefined,
            this.eps
        );
    }

    /**
     * 额外表示信息
     */
    protected extraRepr(): string {
        return `${JSON.stringify(this.normalizedShape)}, eps=${this.eps}, elementwise_affine=${this.elementwiseAffine}`;
    }
}
