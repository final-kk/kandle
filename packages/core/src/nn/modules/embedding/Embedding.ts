/**
 * Embedding - 词嵌入层
 *
 * 将整数索引映射到固定大小的密集向量
 *
 * @see https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
 */

import type { DType } from '@kandle/types';
import { Tensor } from '../../../tensor';
import { Module } from '../../module';
import { Parameter } from '../../parameter';
import { functional } from '../../../generated/ops';
import { empty } from '../../../generated/ops';

/**
 * Embedding 层构造选项
 */
export interface EmbeddingOptions {
    /** 词表大小 */
    numEmbeddings: number;
    /** 嵌入维度 */
    embeddingDim: number;
    /** 填充索引，该位置的嵌入向量将始终为零 */
    paddingIdx?: number;
    /** 数据类型 */
    dtype?: DType;
}

/**
 * Embedding 层
 *
 * 将整数索引映射到固定大小的密集向量
 *
 * @example
 * ```ts
 * // 词表大小 10000，嵌入维度 256
 * const embed = new Embedding(10000, 256);
 * const indices = torch.tensor([1, 2, 3, 4]); // 词索引
 * const vectors = embed.call(indices);         // shape: [4, 256]
 * ```
 */
export class Embedding extends Module {
    /** 词表大小 */
    readonly numEmbeddings: number;

    /** 嵌入维度 */
    readonly embeddingDim: number;

    /** 填充索引 */
    readonly paddingIdx: number | undefined;

    /** 权重矩阵 [num_embeddings, embedding_dim] */
    weight: Parameter;

    /**
     * 创建 Embedding 层
     *
     * @param numEmbeddings 词表大小
     * @param embeddingDim 嵌入维度
     * @param paddingIdx 填充索引，可选
     * @param dtype 数据类型，默认 'float32'
     */
    constructor(numEmbeddings: number, embeddingDim: number, paddingIdx?: number, dtype?: DType);
    constructor(options: EmbeddingOptions);
    constructor(
        arg0: number | EmbeddingOptions,
        arg1?: number,
        arg2?: number,
        arg3?: DType
    ) {
        super();

        let numEmbeddings: number;
        let embeddingDim: number;
        let paddingIdx: number | undefined;
        let dtype: DType = 'float32';

        if (typeof arg0 === 'object') {
            // options 形式
            numEmbeddings = arg0.numEmbeddings;
            embeddingDim = arg0.embeddingDim;
            paddingIdx = arg0.paddingIdx;
            if (arg0.dtype !== undefined) dtype = arg0.dtype;
        } else {
            // 位置参数形式
            numEmbeddings = arg0;
            embeddingDim = arg1 as number;
            paddingIdx = arg2;
            if (arg3 !== undefined) dtype = arg3;
        }

        this.numEmbeddings = numEmbeddings;
        this.embeddingDim = embeddingDim;
        this.paddingIdx = paddingIdx;

        // 创建权重: [num_embeddings, embedding_dim]
        this.weight = new Parameter(
            empty([numEmbeddings, embeddingDim], dtype)
        );
        this.registerParameter('weight', this.weight);

        // 初始化
        this._resetParameters();
    }

    /**
     * 重置参数 (初始化)
     *
     * PyTorch 使用 normal_(0, 1) 初始化
     * 当前简化实现仅供参考
     */
    private _resetParameters(): void {
        // TODO: 实现 normal_ 初始化
        // 如果有 paddingIdx，将该行置零
    }

    /**
     * 前向传播
     *
     * @param input 整数索引张量 [*]
     * @returns 嵌入向量 [*, embedding_dim]
     */
    async forward(input: Tensor): Promise<Tensor> {
        return functional.embedding(
            input,
            this.weight,
            this.paddingIdx
        );
    }

    /**
     * 额外表示信息
     */
    protected extraRepr(): string {
        let s = `${this.numEmbeddings}, ${this.embeddingDim}`;
        if (this.paddingIdx !== undefined) {
            s += `, padding_idx=${this.paddingIdx}`;
        }
        return s;
    }
}
