/**
 * Linear - 线性层 (全连接层)
 *
 * 对输入进行线性变换: y = xW^T + b
 *
 * @see https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
 */

import type { DType } from '@kandle/types';
import { Tensor } from '../../../tensor';
import { Module } from '../../module';
import { Parameter } from '../../parameter';
import { functional } from '../../../generated/ops';
import { zeros, empty } from '../../../generated/ops';

/**
 * Linear 层构造选项
 */
export interface LinearOptions {
    /** 输入特征数 */
    inFeatures: number;
    /** 输出特征数 */
    outFeatures: number;
    /** 是否使用偏置 */
    bias?: boolean;
    /** 数据类型 */
    dtype?: DType;
}

/**
 * Linear 层
 *
 * 对输入进行线性变换: y = xW^T + b
 *
 * @example
 * ```ts
 * const linear = new Linear(10, 5);
 * const input = torch.randn([32, 10]); // batch_size=32, in_features=10
 * const output = linear.call(input);   // shape: [32, 5]
 * ```
 */
export class Linear extends Module {
    /** 输入特征数 */
    readonly inFeatures: number;

    /** 输出特征数 */
    readonly outFeatures: number;

    /** 权重参数 [out_features, in_features] */
    weight: Parameter;

    /** 偏置参数 [out_features]，可选 */
    bias: Parameter | null;

    /**
     * 创建 Linear 层
     *
     * @param inFeatures 输入特征数
     * @param outFeatures 输出特征数
     * @param bias 是否使用偏置，默认 true
     * @param dtype 数据类型，默认 'float32'
     */
    constructor(inFeatures: number, outFeatures: number, bias?: boolean, dtype?: DType);
    constructor(options: LinearOptions);
    constructor(
        arg0: number | LinearOptions,
        arg1?: number | boolean,
        arg2?: boolean,
        arg3?: DType
    ) {
        super();

        let inFeatures: number;
        let outFeatures: number;
        let useBias: boolean = true;
        let dtype: DType = 'float32';

        if (typeof arg0 === 'object') {
            // options 形式
            inFeatures = arg0.inFeatures;
            outFeatures = arg0.outFeatures;
            if (arg0.bias !== undefined) useBias = arg0.bias;
            if (arg0.dtype !== undefined) dtype = arg0.dtype;
        } else {
            // 位置参数形式
            inFeatures = arg0;
            outFeatures = arg1 as number;
            if (typeof arg2 === 'boolean') useBias = arg2;
            if (arg3 !== undefined) dtype = arg3;
        }

        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;

        // 创建权重: [out_features, in_features]
        // 使用 empty 创建未初始化的张量
        // TODO: 实现 kaiming_uniform_ 初始化 (PyTorch 默认)
        this.weight = new Parameter(
            empty([outFeatures, inFeatures], dtype)
        );
        this.registerParameter('weight', this.weight);

        // 创建偏置: [out_features]
        if (useBias) {
            // TODO: 用 uniform(-bound, bound) 初始化
            this.bias = new Parameter(zeros([outFeatures], dtype));
            this.registerParameter('bias', this.bias);
        } else {
            this.bias = null;
            this.registerParameter('bias', null);
        }

        // 初始化权重 (简单实现，后续应使用 kaiming_uniform_)
        this._resetParameters();
    }

    /**
     * 重置参数 (初始化)
     *
     * PyTorch 使用 kaiming_uniform_ 初始化权重
     * 当前简化实现仅供参考
     */
    private _resetParameters(): void {
        // TODO: 实现真正的 kaiming_uniform_ 初始化
        // 当前权重保持未初始化状态
        // 加载预训练权重时会被覆盖
    }

    /**
     * 前向传播
     *
     * @param input 输入张量 [..., in_features]
     * @returns 输出张量 [..., out_features]
     */
    async forward(input: Tensor): Promise<Tensor> {
        return functional.linear(input, this.weight, this.bias ?? undefined);
    }

    /**
     * 额外表示信息
     */
    protected extraRepr(): string {
        return `in_features=${this.inFeatures}, out_features=${this.outFeatures}, bias=${this.bias !== null}`;
    }
}
