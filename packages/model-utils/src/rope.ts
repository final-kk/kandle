/**
 * RoPE (Rotary Position Embedding) 工具
 *
 * 对标 HuggingFace transformers 中的 LlamaRotaryEmbedding
 * 用于 LLaMA3, Qwen3 等现代大语言模型
 *
 * @see https://arxiv.org/abs/2104.09864 RoFormer: Enhanced Transformer with Rotary Position Embedding
 *
 * @example
 * ```ts
 * import { RotaryEmbedding, applyRotaryPosEmbDirect } from '@kandle/model-utils';
 * import { arange } from '@kandle/core';
 *
 * // 创建 RoPE 嵌入器
 * const rope = new RotaryEmbedding({ dim: 64, maxSeqLen: 2048 });
 *
 * // 获取位置嵌入
 * const positions = arange(0, seqLen);
 * const { cos, sin } = rope.forward(positions);
 *
 * // 应用到 Q 和 K
 * const [qRotated, kRotated] = applyRotaryPosEmbDirect(q, k, cos, sin);
 * ```
 *
 * @module @kandle/model-utils/rope
 */

import type { DType } from '@kandle/types';
import {
    Tensor,
    arange,
    cos as torchCos,
    sin as torchSin,
    mul,
    add,
    neg,
    unsqueeze,
    expand,
    slice,
    contiguous,
    cat,
} from '@kandle/core';

// ============================================================================
// Types
// ============================================================================

/**
 * RoPE 配置接口
 */
export interface RopeConfig {
    /**
     * 每个注意力头的维度 (head_dim)
     * 必须是偶数
     */
    dim: number;

    /**
     * 最大序列长度
     * @default 8192
     */
    maxSeqLen?: number;

    /**
     * 基础频率 theta (θ)
     * - LLaMA 2: 10000.0
     * - LLaMA 3 / Qwen3: 500000.0
     * @default 10000.0
     */
    base?: number;

    /**
     * 数据类型
     * @default 'float32'
     */
    dtype?: DType;
}

/**
 * RoPE forward 方法返回值
 */
export interface RopeOutput {
    /** 余弦嵌入 (seq_len, dim) 或 (batch, seq_len, dim) */
    cos: Tensor;
    /** 正弦嵌入 (seq_len, dim) 或 (batch, seq_len, dim) */
    sin: Tensor;
}

// ============================================================================
// RotaryEmbedding Class
// ============================================================================

/**
 * RotaryEmbedding - 旋转位置嵌入
 *
 * 预计算逆频率 (inv_freq)，并根据位置 ID 生成 cos/sin 嵌入。
 *
 * 核心公式:
 * ```
 * inv_freq[i] = 1 / (base ** (2i / dim))  for i in [0, dim/2)
 * freqs = positions @ inv_freq
 * cos_emb = cos(freqs)
 * sin_emb = sin(freqs)
 * ```
 *
 * @example
 * ```ts
 * const rope = new RotaryEmbedding({
 *     dim: 64,        // head_dim
 *     base: 500000,   // LLaMA3 style
 *     maxSeqLen: 8192
 * });
 *
 * // 获取前 128 个位置的嵌入
 * const positions = arange(0, 128); // [128]
 * const { cos, sin } = rope.forward(positions);
 * // cos, sin: [128, 64]
 * ```
 */
export class RotaryEmbedding {
    readonly dim: number;
    readonly maxSeqLen: number;
    readonly base: number;
    readonly dtype: DType;

    // 预计算的逆频率: (dim/2,)
    // 在构造函数中直接初始化，避免在 tidy scope 内创建导致被误释放
    private readonly _invFreq: Tensor;

    constructor(config: RopeConfig) {
        const { dim, maxSeqLen = 8192, base = 10000.0, dtype = 'float32' } = config;

        if (dim % 2 !== 0) {
            throw new Error(`RoPE dim must be even, got ${dim}`);
        }

        this.dim = dim;
        this.maxSeqLen = maxSeqLen;
        this.base = base;
        this.dtype = dtype;

        // 直接初始化逆频率张量
        this._invFreq = this._computeInvFreqDirect();
    }

    /**
     * 获取逆频率张量
     *
     * inv_freq[i] = 1.0 / (base ** (2i / dim))
     * shape: (dim/2,)
     */
    get invFreq(): Tensor {
        return this._invFreq;
    }

    /**
     * 直接计算逆频率 (使用标量运算避免复杂的张量操作)
     */
    private _computeInvFreqDirect(): Tensor {
        const halfDim = this.dim / 2;
        const values = new Float32Array(halfDim);

        for (let i = 0; i < halfDim; i++) {
            // inv_freq[i] = 1.0 / (base ** (2 * i / dim))
            // = 1.0 / (base ** (i / halfDim))
            const exponent = i / halfDim;
            values[i] = 1.0 / Math.pow(this.base, exponent);
        }

        return new Tensor(values, { dtype: this.dtype, shape: [halfDim] });
    }

    /**
     * 正向传播 - 计算旋转位置嵌入
     *
     * @param positions 位置索引张量，形状 (seq_len,) 或 (batch, seq_len)
     * @returns 包含 cos 和 sin 的对象
     *
     * @example
     * ```ts
     * // 单序列
     * const positions = arange(0, 128);  // [128]
     * const { cos, sin } = rope.forward(positions);
     * // cos, sin: [128, 64]
     *
     * // 批量 (不同序列可能有不同的起始位置)
     * const positions = tensor([[0,1,2,3], [10,11,12,13]]);  // [2, 4]
     * const { cos, sin } = rope.forward(positions);
     * // cos, sin: [2, 4, 64]
     * ```
     */
    forward(positions: Tensor): RopeOutput {
        const invFreq = this.invFreq; // (halfDim,)

        // 获取位置张量的形状
        const posShape = positions.shape;
        const isBatched = posShape.length === 2;

        // 扩展 inv_freq 以便广播
        // positions: (..., seqLen)
        // inv_freq: (halfDim,)
        // 需要 positions @ inv_freq.T 或使用广播乘法

        let freqs: Tensor;

        if (isBatched) {
            // positions: (batch, seqLen) -> (batch, seqLen, 1)
            // inv_freq: (halfDim,) -> (1, 1, halfDim)
            // freqs = positions[..., None] * inv_freq[None, None, :]
            const batchSize = posShape[0];
            const seqLen = posShape[1];

            // positions: (batch, seqLen) -> (batch, seqLen, 1)
            const posExpanded = unsqueeze(positions, -1);

            // inv_freq: (halfDim,) -> (1, 1, halfDim)
            let invFreqExpanded = unsqueeze(invFreq, 0);
            invFreqExpanded = unsqueeze(invFreqExpanded, 0);
            invFreqExpanded = expand(invFreqExpanded, [batchSize, seqLen, this.dim / 2]);

            // posExpanded: (batch, seqLen, 1)
            // 扩展 posExpanded 到 (batch, seqLen, halfDim)
            const posExpandedFull = expand(posExpanded, [batchSize, seqLen, this.dim / 2]);

            // freqs = positions * inv_freq
            freqs = mul(posExpandedFull, invFreqExpanded);
        } else {
            // positions: (seqLen,) -> (seqLen, 1)
            // inv_freq: (halfDim,) -> (1, halfDim)
            const seqLen = posShape[0];

            // positions: (seqLen,) -> (seqLen, 1)
            const posExpanded = unsqueeze(positions, -1);

            // inv_freq: (halfDim,) -> (1, halfDim)
            let invFreqExpanded = unsqueeze(invFreq, 0);
            invFreqExpanded = expand(invFreqExpanded, [seqLen, this.dim / 2]);

            // posExpanded: (seqLen, 1
            // 扩展到 (seqLen, halfDim)
            const posExpandedFull = expand(posExpanded, [seqLen, this.dim / 2]);

            // freqs = positions * inv_freq
            freqs = mul(posExpandedFull, invFreqExpanded);
        }

        // freqs: (..., halfDim)
        // 对标 HuggingFace: emb = torch.cat((freqs, freqs), dim=-1)
        // 然后 cos = emb.cos(), sin = emb.sin()
        // 这样 cos/sin 的形状是 (..., dim) 而不是 (..., halfDim)
        const emb = cat([freqs, freqs], -1);  // (..., dim)

        // 计算 cos 和 sin
        const cosFreqs = torchCos(emb);
        const sinFreqs = torchSin(emb);

        return {
            cos: cosFreqs,
            sin: sinFreqs,
        };
    }

    /**
     * 获取指定序列长度的位置 ID
     *
     * @param seqLen 序列长度
     * @param startPos 起始位置 (用于增量解码)
     * @returns 位置张量 (seqLen,)
     */
    getPositionIds(seqLen: number, startPos: number = 0): Tensor {
        return arange(startPos, startPos + seqLen, 1, 'int32');
    }
}

// ============================================================================
// Functional API
// ============================================================================

/**
 * 旋转半维度 (rotate_half)
 *
 * 将输入的后半部分维度取反并移到前面
 *
 * 公式: [-x2, x1] where x = [x1, x2]
 *
 * @param x 输入张量，形状 (..., dim)，dim 必须是偶数
 * @returns 旋转后的张量，形状 (..., dim)
 *
 * @example
 * ```ts
 * const x = tensor([[1, 2, 3, 4]]);  // [1, 4]
 * const rotated = rotateHalf(x);
 * // rotated: [[-3, -4, 1, 2]]
 * ```
 */
export function rotateHalf(x: Tensor): Tensor {
    const shape = x.shape;
    const dim = shape[shape.length - 1];

    if (dim % 2 !== 0) {
        throw new Error(`rotateHalf requires even last dimension, got ${dim}`);
    }

    const halfDim = dim / 2;

    // x1 = x[..., :halfDim]
    // x2 = x[..., halfDim:]
    // 使用切片操作
    const sliceStr1 = `..., :${halfDim}`;
    const sliceStr2 = `..., ${halfDim}:`;

    const x1 = slice(x, sliceStr1);
    const x2 = slice(x, sliceStr2);

    // -x2
    const negX2 = neg(x2);

    // 构造 [-x2, x1]
    // 由于没有 cat，我们需要手动实现
    // 使用 empty 创建输出，然后使用 scatter 或类似操作填充
    // 但这很复杂...

    // 替代方案：使用加法技巧
    // 创建两个张量，一个在前半部分有值，一个在后半部分有值
    // 然后相加

    // 实际上最简洁的方式是直接返回两部分，在 applyRotaryPosEmb 中处理
    // 但为了兼容标准 API，我们需要 cat

    // 临时解决方案：如果没有 cat，我们可以创建一个新的数据数组
    // 这需要 dataAsync，这是异步的...

    // 更好的方案：直接在 applyRotaryPosEmb 中使用分开的计算
    // 而不是使用标准的 rotate_half

    // 对于现在，我们抛出一个待实现错误，并提供一个替代实现
    return rotateHalfAlternative(x);
}

/**
 * rotate_half 的替代实现 (避免使用 cat)
 *
 * 使用 view 重组维度，然后交换并取反
 * 原理：将 (..., dim) reshape 为 (..., 2, dim/2)，交换最后两组，取反后再 reshape 回来
 */
function rotateHalfAlternative(x: Tensor): Tensor {
    const shape = x.shape;
    const ndim = shape.length;
    const dim = shape[ndim - 1];
    const halfDim = dim / 2;

    // 方法：利用数学等价关系
    // rotate_half 的核心是: [-x2, x1]
    // 如果我们能用 permute + neg 的组合来实现...

    // 另一个思路：reshape to (..., 2, halfDim), 然后 permute + neg
    // 但 permute 不能只交换子部分...

    // 实际最简单的方式：直接操作数据
    // 但这违反了纯张量运算原则

    // 最终方案：使用 slice + view + 逐元素运算
    // 由于 slice 返回视图，我们可以：
    // 1. x1 = x[..., :halfDim]  -> (*, halfDim)
    // 2. x2 = x[..., halfDim:]  -> (*, halfDim)
    // 3. 结果应该是 [-x2, x1]

    // 如果没有 cat，我们可以利用广播和 where 操作
    // 或者创建一个掩码...

    // 实际上，让我们采用最实用的方案:
    // 在内部使用 JS 数组操作来实现 cat
    // 这对于推理场景是可接受的

    // 暂时使用循环实现，后续可以优化为内核
    throw new Error(
        'rotateHalf is not yet implemented due to missing cat operation. ' +
        'Use applyRotaryPosEmbDirect which avoids cat.'
    );
}

/**
 * 应用旋转位置嵌入到 Query 和 Key
 *
 * 公式:
 * ```
 * q_rotated = q * cos + rotate_half(q) * sin
 * k_rotated = k * cos + rotate_half(k) * sin
 * ```
 *
 * @param q Query 张量，形状 (batch, heads, seq, head_dim)
 * @param k Key 张量，形状 (batch, heads, seq, head_dim)
 * @param cos 余弦嵌入，形状 (seq, head_dim) 或 (batch, seq, head_dim) 或 (batch, seq, head_dim/2)
 * @param sin 正弦嵌入，形状同 cos
 * @param unsqueezeDim 在哪个维度插入以便广播，默认 1 (heads 维度)
 * @returns [q_rotated, k_rotated]
 */
export function applyRotaryPosEmb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    unsqueezeDim: number = 1
): [Tensor, Tensor] {
    // 检查 cos/sin 的维度
    // 如果是 half_dim，需要扩展到 full_dim

    const qShape = q.shape;
    const headDim = qShape[qShape.length - 1];
    const cosShape = cos.shape;
    const cosDim = cosShape[cosShape.length - 1];

    if (cosDim === headDim / 2) {
        // 需要扩展：使用 applyRotaryPosEmbHalfDim
        return applyRotaryPosEmbHalfDim(q, k, cos, sin, unsqueezeDim);
    }

    // cos/sin: (seq, head_dim) 或 (batch, seq, head_dim)
    // 需要 unsqueeze 到 heads 维度

    let cosExpanded = cos;
    let sinExpanded = sin;

    // 如果 cos/sin 是 2D (seq, head_dim)，需要 unsqueeze 到 (1, 1, seq, head_dim)
    // 如果是 3D (batch, seq, head_dim)，需要 unsqueeze 到 (batch, 1, seq, head_dim)
    if (cos.shape.length === 2) {
        // (seq, head_dim) -> (1, 1, seq, head_dim)
        cosExpanded = unsqueeze(cos, 0);
        cosExpanded = unsqueeze(cosExpanded, 0);
        sinExpanded = unsqueeze(sin, 0);
        sinExpanded = unsqueeze(sinExpanded, 0);
    } else if (cos.shape.length === 3) {
        // (batch, seq, head_dim) -> (batch, 1, seq, head_dim)
        cosExpanded = unsqueeze(cos, unsqueezeDim);
        sinExpanded = unsqueeze(sin, unsqueezeDim);
    }

    // 应用旋转嵌入
    // q_embed = q * cos + rotate_half(q) * sin
    // k_embed = k * cos + rotate_half(k) * sin

    const qEmbed = add(mul(q, cosExpanded), mul(rotateHalf(q), sinExpanded));
    const kEmbed = add(mul(k, cosExpanded), mul(rotateHalf(k), sinExpanded));

    return [qEmbed, kEmbed];
}

/**
 * 应用旋转位置嵌入 (half_dim 版本)
 *
 * 当 cos/sin 的最后维度是 head_dim/2 时使用
 * 这是 RotaryEmbedding.forward 返回的格式
 */
function applyRotaryPosEmbHalfDim(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    unsqueezeDim: number = 1
): [Tensor, Tensor] {
    // q/k: (batch, heads, seq, head_dim)
    // cos/sin: (..., head_dim/2)

    const qShape = q.shape;
    const headDim = qShape[qShape.length - 1];
    const halfDim = headDim / 2;

    // 分割 q 和 k 为两半
    const sliceStrFirst = `..., :${halfDim}`;
    const sliceStrSecond = `..., ${halfDim}:`;

    const q1 = slice(q, sliceStrFirst);
    const q2 = slice(q, sliceStrSecond);
    const k1 = slice(k, sliceStrFirst);
    const k2 = slice(k, sliceStrSecond);

    // 扩展 cos/sin 以便广播
    let cosExpanded = cos;
    let sinExpanded = sin;

    if (cos.shape.length === 2) {
        // (seq, halfDim) -> (1, 1, seq, halfDim)
        cosExpanded = unsqueeze(cos, 0);
        cosExpanded = unsqueeze(cosExpanded, 0);
        sinExpanded = unsqueeze(sin, 0);
        sinExpanded = unsqueeze(sinExpanded, 0);
    } else if (cos.shape.length === 3) {
        // (batch, seq, halfDim) -> (batch, 1, seq, halfDim)
        cosExpanded = unsqueeze(cos, unsqueezeDim);
        sinExpanded = unsqueeze(sin, unsqueezeDim);
    }

    // 应用旋转:
    // 对于 [q1, q2]:
    // result[..., :halfDim] = q1 * cos - q2 * sin
    // result[..., halfDim:] = q2 * cos + q1 * sin

    // q_embed_1 = q1 * cos - q2 * sin
    // q_embed_2 = q2 * cos + q1 * sin
    const qEmbed1 = add(mul(q1, cosExpanded), mul(neg(q2), sinExpanded));
    const qEmbed2 = add(mul(q2, cosExpanded), mul(q1, sinExpanded));

    const kEmbed1 = add(mul(k1, cosExpanded), mul(neg(k2), sinExpanded));
    const kEmbed2 = add(mul(k2, cosExpanded), mul(k1, sinExpanded));

    // 需要 cat [qEmbed1, qEmbed2] 在最后一个维度
    // 由于没有 cat，我们使用备用方案

    // 备用方案：创建完整输出并填充
    // 但这需要可变操作或 cat...

    // 最终方案：返回重组后的张量
    // 由于 slice 返回视图，我们可以利用这一点
    // 但实际上没有原地写入能力...

    // 临时解决方案：抛出错误并建议使用全维度 cos/sin
    throw new Error(
        'applyRotaryPosEmbHalfDim requires cat operation which is not yet implemented. ' +
        'Please use full-dimension cos/sin from a modified RotaryEmbedding.'
    );
}

/**
 * 直接应用旋转位置嵌入 (避免使用 cat)
 *
 * 这是一个优化版本，直接操作半维度，避免使用 cat 操作。
 * 适用于推理场景。
 *
 * @param q Query 张量，形状 (batch, heads, seq, head_dim)
 * @param k Key 张量，形状 (batch, heads, seq, head_dim)
 * @param cos 余弦嵌入，形状 (..., head_dim/2)
 * @param sin 正弦嵌入，形状 (..., head_dim/2)
 * @param unsqueezeDim 在哪个维度插入以便广播，默认 1
 * @returns [q_rotated, k_rotated]
 *
 * @example
 * ```ts
 * const rope = new RotaryEmbedding({ dim: 64 });
 * const positions = arange(0, seqLen);
 * const { cos, sin } = rope.forward(positions);
 *
 * // Q/K shape: (batch, heads, seq, head_dim)
 * const [qRot, kRot] = applyRotaryPosEmbDirect(q, k, cos, sin);
 * ```
 */
export function applyRotaryPosEmbDirect(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    unsqueezeDim: number = 1
): [Tensor, Tensor] {
    // 对标 HuggingFace 的实现:
    // q_embed = (q * cos) + (rotate_half(q) * sin)
    // 其中 rotate_half(x) = cat(-x2, x1, dim=-1)

    const qShape = q.shape;
    const headDim = qShape[qShape.length - 1];

    // 验证 cos/sin 维度
    const cosLastDim = cos.shape[cos.shape.length - 1];
    if (cosLastDim !== headDim) {
        throw new Error(
            `cos/sin last dimension (${cosLastDim}) must equal head_dim (${headDim}). ` +
            `Make sure RotaryEmbedding.forward() returns full-dimension cos/sin.`
        );
    }

    // 扩展 cos/sin 以便广播 - 这些是 view，不需要 dispose
    let cosExpanded = cos;
    let sinExpanded = sin;
    const intermediates: Tensor[] = [];

    if (cos.shape.length === 2) {
        // (seq, head_dim) -> (1, 1, seq, head_dim)
        const cos1 = unsqueeze(cos, 0);
        cosExpanded = unsqueeze(cos1, 0);
        intermediates.push(cos1);

        const sin1 = unsqueeze(sin, 0);
        sinExpanded = unsqueeze(sin1, 0);
        intermediates.push(sin1);
    } else if (cos.shape.length === 3) {
        // (batch, seq, head_dim) -> (batch, 1, seq, head_dim)
        cosExpanded = unsqueeze(cos, unsqueezeDim);
        sinExpanded = unsqueeze(sin, unsqueezeDim);
    }

    // rotate_half: 返回结果和需要释放的中间张量
    const rotateHalfTensor = (x: Tensor): { result: Tensor; toDispose: Tensor[] } => {
        const xShape = x.shape;
        const dim = xShape[xShape.length - 1];
        const halfDim = dim / 2;

        const sliceStrFirst = `..., :${halfDim}`;
        const sliceStrSecond = `..., ${halfDim}:`;

        const x1Slice = slice(x, sliceStrFirst);
        const x2Slice = slice(x, sliceStrSecond);

        // contiguous 创建新张量
        const x1 = contiguous(x1Slice);
        const x2 = contiguous(x2Slice);

        // neg 创建新张量
        const negX2 = neg(x2);

        // cat 创建新张量
        const result = cat([negX2, x1], -1);

        // 返回结果和需要释放的中间张量
        // x1Slice, x2Slice 是 view，不需要释放
        return { result, toDispose: [x1, x2, negX2] };
    };

    // 计算 rotate_half
    const qRotHalf = rotateHalfTensor(q);
    const kRotHalf = rotateHalfTensor(k);

    // 计算 q * cos 和 k * cos
    const qMulCos = mul(q, cosExpanded);
    const kMulCos = mul(k, cosExpanded);

    // 计算 rotate_half * sin
    const qRotMulSin = mul(qRotHalf.result, sinExpanded);
    const kRotMulSin = mul(kRotHalf.result, sinExpanded);

    // 最终结果
    const qRotated = add(qMulCos, qRotMulSin);
    const kRotated = add(kMulCos, kRotMulSin);

    // 释放所有中间张量
    for (const t of intermediates) t.dispose();
    for (const t of qRotHalf.toDispose) t.dispose();
    for (const t of kRotHalf.toDispose) t.dispose();
    qRotHalf.result.dispose();
    kRotHalf.result.dispose();
    qMulCos.dispose();
    kMulCos.dispose();
    qRotMulSin.dispose();
    kRotMulSin.dispose();

    return [qRotated, kRotated];
}

/**
 * 手动实现 tensor concatenation (临时方案)
 *
 * 由于框架尚未提供 cat 操作，这里使用同步数据操作实现。
 * 注意：这会触发数据同步，可能影响性能。
 *
 * @param a 第一个张量
 * @param b 第二个张量
 * @param dim 拼接维度 (支持负数)
 * @returns 拼接后的张量
 */
function catTensors(a: Tensor, b: Tensor, dim: number): Tensor {
    const aShape = a.shape;
    const bShape = b.shape;

    // 规范化 dim
    const ndim = aShape.length;
    const normDim = dim < 0 ? ndim + dim : dim;

    if (normDim < 0 || normDim >= ndim) {
        throw new Error(`Invalid dimension ${dim} for tensor with ${ndim} dimensions`);
    }

    // 检查形状兼容性
    for (let i = 0; i < ndim; i++) {
        if (i !== normDim && aShape[i] !== bShape[i]) {
            throw new Error(
                `Shapes are not compatible for concatenation: ` +
                `${aShape} and ${bShape} at dimension ${normDim}`
            );
        }
    }

    // 计算输出形状
    const outShape = [...aShape];
    outShape[normDim] = aShape[normDim] + bShape[normDim];

    // 获取数据 (同步操作)
    const aData = a.data;
    const bData = b.data;

    // 计算 strides
    const computeStrides = (shape: readonly number[]): number[] => {
        const strides = new Array(shape.length);
        strides[shape.length - 1] = 1;
        for (let i = shape.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    };

    const outStrides = computeStrides(outShape);
    const aStrides = computeStrides(aShape as number[]);
    const bStrides = computeStrides(bShape as number[]);

    // 创建输出数据
    const outSize = outShape.reduce((acc, v) => acc * v, 1);
    const outData = new Float32Array(outSize);

    // 复制数据
    const copyRecursive = (
        indices: number[],
        currentDim: number,
        aOffset: number,
        bOffset: number,
        outOffset: number
    ) => {
        if (currentDim === ndim) {
            // 到达叶子
            outData[outOffset] = indices[normDim] < aShape[normDim]
                ? Number(aData[aOffset])
                : Number(bData[bOffset]);
            return;
        }

        const maxIdx = outShape[currentDim];
        const aSize = aShape[currentDim];

        for (let i = 0; i < maxIdx; i++) {
            indices[currentDim] = i;

            let newAOffset = aOffset;
            let newBOffset = bOffset;

            if (currentDim === normDim) {
                if (i < aSize) {
                    newAOffset = aOffset + i * aStrides[currentDim];
                } else {
                    newBOffset = bOffset + (i - aSize) * bStrides[currentDim];
                }
            } else {
                newAOffset = aOffset + i * aStrides[currentDim];
                newBOffset = bOffset + i * bStrides[currentDim];
            }

            copyRecursive(
                indices,
                currentDim + 1,
                newAOffset,
                newBOffset,
                outOffset + i * outStrides[currentDim]
            );
        }
    };

    // 使用更简单的线性复制方法
    // 对于最后一个维度的拼接，这更高效
    if (normDim === ndim - 1) {
        // 沿最后一个维度拼接
        const batchSize = outShape.slice(0, -1).reduce((acc, v) => acc * v, 1);
        const aDim = aShape[normDim];
        const bDim = bShape[normDim];
        const outDim = outShape[normDim];

        for (let batch = 0; batch < batchSize; batch++) {
            const aStart = batch * aDim;
            const bStart = batch * bDim;
            const outStart = batch * outDim;

            // 复制 a 的部分
            for (let i = 0; i < aDim; i++) {
                outData[outStart + i] = Number(aData[aStart + i]);
            }
            // 复制 b 的部分
            for (let i = 0; i < bDim; i++) {
                outData[outStart + aDim + i] = Number(bData[bStart + i]);
            }
        }
    } else {
        // 通用情况
        copyRecursive(new Array(ndim).fill(0), 0, 0, 0, 0);
    }

    return new Tensor(outData, { dtype: a.dtype, shape: outShape });
}

// ============================================================================
// Exports
// ============================================================================

// 原生 cat 实现现在从 generated/ops 导入，catTensors 保留作为内部后备
// export { catTensors as cat };  // 已废弃，使用 generated/ops.cat
