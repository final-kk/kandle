/**
 * Sinusoidal Positional Encoding
 *
 * 对标原始 Transformer 论文 "Attention Is All You Need" 的位置编码
 * @see https://arxiv.org/abs/1706.03762
 *
 * 数学公式:
 * PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
 * PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
 *
 * 特点:
 * - 不需要学习参数，完全确定性
 * - 可以外推到训练时未见过的序列长度
 * - 相对位置信息通过三角函数的性质隐式编码
 *
 * @example
 * ```ts
 * import { SinusoidalPositionalEncoding } from '@kandle/model-utils';
 *
 * // 创建位置编码器
 * const pe = new SinusoidalPositionalEncoding({ dModel: 512 });
 *
 * // 方式1: 直接获取位置编码
 * const encoding = pe.forward(128);  // (128, 512)
 *
 * // 方式2: 应用到输入 embedding
 * const x = embeddings;  // (batch, seqLen, dModel)
 * const xWithPE = pe.apply(x);  // (batch, seqLen, dModel)
 * ```
 *
 * @module @kandle/model-utils/sinusoidal
 */

import type { DType } from '@kandle/types';
import {
    Tensor,
    add,
    unsqueeze,
    expand,
} from '@kandle/core';

// ============================================================================
// Types
// ============================================================================

/**
 * 正弦位置编码配置接口
 */
export interface SinusoidalConfig {
    /**
     * 模型维度 (d_model)
     * 必须是偶数 (以交替使用 sin 和 cos)
     */
    dModel: number;

    /**
     * 最大序列长度
     * 预计算到此长度以提高效率
     * @default 8192
     */
    maxSeqLen?: number;

    /**
     * 基础频率 (base frequency)
     * 原始 Transformer 论文使用 10000
     * @default 10000.0
     */
    base?: number;

    /**
     * 数据类型
     * @default 'float32'
     */
    dtype?: DType;
}

// ============================================================================
// SinusoidalPositionalEncoding Class
// ============================================================================

/**
 * SinusoidalPositionalEncoding - 正弦位置编码
 *
 * 实现原始 Transformer 论文中的位置编码方案:
 * - 偶数维度使用 sin
 * - 奇数维度使用 cos
 * - 频率随着维度呈指数递减
 *
 * 核心公式:
 * ```
 * div_term[i] = 1 / (base ** (2i / d_model))  for i in [0, d_model/2)
 *
 * PE[pos, 2i]   = sin(pos * div_term[i])
 * PE[pos, 2i+1] = cos(pos * div_term[i])
 * ```
 *
 * 与 RoPE 的区别:
 * - 正弦位置编码是加性的 (x + PE)
 * - RoPE 是乘性的 (旋转 x)
 *
 * 使用场景:
 * - Whisper encoder/decoder
 * - 原始 Transformer
 * - BERT (虽然 BERT 通常用可学习位置编码)
 * - GPT-1
 *
 * @example
 * ```ts
 * const pe = new SinusoidalPositionalEncoding({
 *     dModel: 512,
 *     maxSeqLen: 4096
 * });
 *
 * // 获取位置编码
 * const encoding = pe.forward(128);  // shape: (128, 512)
 *
 * // 应用到输入
 * const embeddings = model.embed(inputIds);  // (batch, seq, 512)
 * const withPE = pe.apply(embeddings);       // (batch, seq, 512)
 * ```
 */
export class SinusoidalPositionalEncoding {
    readonly dModel: number;
    readonly maxSeqLen: number;
    readonly base: number;
    readonly dtype: DType;

    // 预计算的位置编码: (maxSeqLen, dModel)
    private _encodings: Tensor | null = null;
    // 保存原始数据用于切片操作 (避免 GPU 同步问题)
    private _rawData: Float32Array | null = null;

    constructor(config: SinusoidalConfig) {
        const { dModel, maxSeqLen = 8192, base = 10000.0, dtype = 'float32' } = config;

        if (dModel % 2 !== 0) {
            throw new Error(`SinusoidalPositionalEncoding requires even dModel, got ${dModel}`);
        }

        if (dModel <= 0) {
            throw new Error(`dModel must be positive, got ${dModel}`);
        }

        if (maxSeqLen <= 0) {
            throw new Error(`maxSeqLen must be positive, got ${maxSeqLen}`);
        }

        this.dModel = dModel;
        this.maxSeqLen = maxSeqLen;
        this.base = base;
        this.dtype = dtype;
    }

    /**
     * 获取预计算的位置编码 (延迟初始化)
     *
     * shape: (maxSeqLen, dModel)
     */
    get encodings(): Tensor {
        if (this._encodings === null) {
            this._computeEncodings();
        }
        return this._encodings!;
    }

    /**
     * 获取原始数据数组 (用于切片)
     */
    private get rawData(): Float32Array {
        if (this._rawData === null) {
            this._computeEncodings();
        }
        return this._rawData!;
    }

    /**
     * 计算位置编码
     *
     * 实现细节:
     * 1. 计算 div_term: 频率递减因子
     * 2. 对每个位置计算 sin/cos
     * 3. 交替排列 sin 和 cos
     */
    private _computeEncodings(): void {
        const { dModel, maxSeqLen, base, dtype } = this;
        const halfDim = dModel / 2;

        // 创建输出数组 (maxSeqLen, dModel)
        const data = new Float32Array(maxSeqLen * dModel);

        // 预计算 div_term: 1 / (base ** (2i / d_model)) for i in [0, halfDim)
        // 等价于 exp(-2i * log(base) / d_model)
        const divTerm = new Float32Array(halfDim);
        const logBase = Math.log(base);
        for (let i = 0; i < halfDim; i++) {
            // div_term[i] = 1 / (base ** (2i / d_model))
            // = exp(-2i * log(base) / d_model)
            divTerm[i] = Math.exp(-2 * i * logBase / dModel);
        }

        // 对每个位置计算编码
        for (let pos = 0; pos < maxSeqLen; pos++) {
            for (let i = 0; i < halfDim; i++) {
                const angle = pos * divTerm[i];

                // PE[pos, 2i] = sin(angle)
                data[pos * dModel + 2 * i] = Math.sin(angle);

                // PE[pos, 2i + 1] = cos(angle)
                data[pos * dModel + 2 * i + 1] = Math.cos(angle);
            }
        }

        // 保存原始数据用于切片操作
        this._rawData = data;
        // 创建 Tensor
        this._encodings = new Tensor(data, { dtype, shape: [maxSeqLen, dModel] });
    }

    /**
     * 获取指定长度的位置编码
     *
     * @param seqLen 序列长度
     * @returns 位置编码张量 (seqLen, dModel)
     *
     * @throws Error 如果 seqLen > maxSeqLen
     *
     * @example
     * ```ts
     * const pe = new SinusoidalPositionalEncoding({ dModel: 512 });
     * const encoding = pe.forward(128);  // (128, 512)
     * ```
     */
    forward(seqLen: number): Tensor {
        if (seqLen > this.maxSeqLen) {
            throw new Error(
                `Sequence length ${seqLen} exceeds maxSeqLen ${this.maxSeqLen}. ` +
                `Either reduce sequence length or create a new SinusoidalPositionalEncoding ` +
                `with larger maxSeqLen.`
            );
        }

        if (seqLen <= 0) {
            throw new Error(`seqLen must be positive, got ${seqLen}`);
        }

        // 如果请求的长度正好是 maxSeqLen，直接返回
        if (seqLen === this.maxSeqLen) {
            return this.encodings;
        }

        // 否则切片返回前 seqLen 个位置
        // 使用 slice 操作
        return this._sliceEncodings(seqLen);
    }

    /**
     * 切片位置编码
     *
     * 使用缓存的原始数据，避免 GPU 同步问题
     */
    private _sliceEncodings(seqLen: number): Tensor {
        const data = this.rawData;

        // 提取前 seqLen 行
        const slicedData = new Float32Array(seqLen * this.dModel);
        for (let pos = 0; pos < seqLen; pos++) {
            for (let d = 0; d < this.dModel; d++) {
                slicedData[pos * this.dModel + d] = data[pos * this.dModel + d];
            }
        }

        return new Tensor(slicedData, { dtype: this.dtype, shape: [seqLen, this.dModel] });
    }

    /**
     * 应用位置编码到输入
     *
     * 执行操作: output = x + PE[:seqLen]
     *
     * @param x 输入张量，形状 (batch, seqLen, dModel) 或 (seqLen, dModel)
     * @returns 添加位置编码后的张量，形状与输入相同
     *
     * @throws Error 如果输入最后一维不等于 dModel
     * @throws Error 如果 seqLen > maxSeqLen
     *
     * @example
     * ```ts
     * const pe = new SinusoidalPositionalEncoding({ dModel: 512 });
     *
     * // 批量输入
     * const embeddings = embed(inputIds);  // (2, 128, 512)
     * const withPE = pe.apply(embeddings); // (2, 128, 512)
     *
     * // 单序列输入 (不常见)
     * const single = embed(inputIds);      // (128, 512)
     * const singleWithPE = pe.apply(single); // (128, 512)
     * ```
     */
    apply(x: Tensor): Tensor {
        const xShape = x.shape;
        const ndim = xShape.length;

        if (ndim < 2) {
            throw new Error(
                `Input tensor must have at least 2 dimensions, got ${ndim}. ` +
                `Expected shape: (batch, seqLen, dModel) or (seqLen, dModel)`
            );
        }

        const lastDim = xShape[ndim - 1];
        if (lastDim !== this.dModel) {
            throw new Error(
                `Input last dimension (${lastDim}) must equal dModel (${this.dModel})`
            );
        }

        const seqLen = xShape[ndim - 2];
        if (seqLen > this.maxSeqLen) {
            throw new Error(
                `Sequence length ${seqLen} exceeds maxSeqLen ${this.maxSeqLen}`
            );
        }

        // 获取对应长度的位置编码
        const pe = this.forward(seqLen);  // (seqLen, dModel)

        if (ndim === 2) {
            // 输入: (seqLen, dModel)
            // 位置编码: (seqLen, dModel)
            // 直接相加
            return add(x, pe);
        } else {
            // 输入: (batch, seqLen, dModel) 或更多维度
            // 位置编码: (seqLen, dModel) -> 需要广播
            // 扩展 pe 到 (1, ..., 1, seqLen, dModel)

            let peExpanded = pe;

            // 在前面添加维度以匹配 batch 维度
            // 例如 (seqLen, dModel) -> (1, seqLen, dModel) for 3D input
            const numBatchDims = ndim - 2;
            for (let i = 0; i < numBatchDims; i++) {
                peExpanded = unsqueeze(peExpanded, 0);
            }

            // 使用 expand 广播到完整形状
            const targetShape = [...xShape] as number[];
            peExpanded = expand(peExpanded, targetShape);

            return add(x, peExpanded);
        }
    }

    /**
     * 获取指定位置范围的编码
     *
     * 用于增量解码时获取连续位置的编码
     *
     * @param startPos 起始位置 (包含)
     * @param endPos 结束位置 (不包含)
     * @returns 位置编码张量 (endPos - startPos, dModel)
     *
     * @example
     * ```ts
     * const pe = new SinusoidalPositionalEncoding({ dModel: 512 });
     *
     * // 首次解码: 位置 0-10
     * const pe0 = pe.getRange(0, 10);  // (10, 512)
     *
     * // 增量解码: 位置 10-11
     * const pe1 = pe.getRange(10, 11); // (1, 512)
     * ```
     */
    getRange(startPos: number, endPos: number): Tensor {
        if (startPos < 0) {
            throw new Error(`startPos must be non-negative, got ${startPos}`);
        }
        if (endPos <= startPos) {
            throw new Error(`endPos (${endPos}) must be greater than startPos (${startPos})`);
        }
        if (endPos > this.maxSeqLen) {
            throw new Error(
                `endPos ${endPos} exceeds maxSeqLen ${this.maxSeqLen}`
            );
        }

        const data = this.rawData;
        const length = endPos - startPos;

        // 提取 [startPos, endPos) 的行
        const slicedData = new Float32Array(length * this.dModel);
        for (let i = 0; i < length; i++) {
            const pos = startPos + i;
            for (let d = 0; d < this.dModel; d++) {
                slicedData[i * this.dModel + d] = data[pos * this.dModel + d];
            }
        }

        return new Tensor(slicedData, { dtype: this.dtype, shape: [length, this.dModel] });
    }

    /**
     * 释放预计算的编码 (用于内存管理)
     *
     * 下次调用 forward/apply/getRange 时会重新计算
     */
    clear(): void {
        this._encodings = null;
        this._rawData = null;
    }
}

// ============================================================================
// Functional API
// ============================================================================

/**
 * 创建正弦位置编码 (函数式接口)
 *
 * 这是一个便捷函数,适用于只需要一次性获取编码的场景。
 * 如果需要重复获取编码,建议使用 SinusoidalPositionalEncoding 类以复用预计算的编码。
 *
 * @param seqLen 序列长度
 * @param dModel 模型维度
 * @param base 基础频率 (默认 10000)
 * @param dtype 数据类型 (默认 'float32')
 * @returns 位置编码张量 (seqLen, dModel)
 *
 * @example
 * ```ts
 * const pe = createSinusoidalEncoding(128, 512);  // (128, 512)
 * const output = add(embeddings, pe);
 * ```
 */
export function createSinusoidalEncoding(
    seqLen: number,
    dModel: number,
    base: number = 10000.0,
    dtype: DType = 'float32'
): Tensor {
    // 验证参数
    if (dModel % 2 !== 0) {
        throw new Error(`dModel must be even, got ${dModel}`);
    }
    if (seqLen <= 0) {
        throw new Error(`seqLen must be positive, got ${seqLen}`);
    }
    if (dModel <= 0) {
        throw new Error(`dModel must be positive, got ${dModel}`);
    }

    const halfDim = dModel / 2;

    // 创建输出数组
    const data = new Float32Array(seqLen * dModel);

    // 预计算 div_term
    const logBase = Math.log(base);
    const divTerm = new Float32Array(halfDim);
    for (let i = 0; i < halfDim; i++) {
        divTerm[i] = Math.exp(-2 * i * logBase / dModel);
    }

    // 计算编码
    for (let pos = 0; pos < seqLen; pos++) {
        for (let i = 0; i < halfDim; i++) {
            const angle = pos * divTerm[i];
            data[pos * dModel + 2 * i] = Math.sin(angle);
            data[pos * dModel + 2 * i + 1] = Math.cos(angle);
        }
    }

    return new Tensor(data, { dtype, shape: [seqLen, dModel] });
}

// ============================================================================
// Exports
// ============================================================================

// 类型导出在顶层声明处
