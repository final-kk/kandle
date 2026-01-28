/**
 * μ-law 编解码
 *
 * 对标 torchaudio.functional.mu_law_encoding / mu_law_decoding
 *
 * 用于语音压缩和量化的经典算法
 */

import { type Tensor } from '../../tensor';
import * as k from '../../index';

/**
 * μ-law 编码
 *
 * 将 [-1, 1] 范围的信号压缩编码为 [0, quantization_channels - 1] 的整数
 *
 * 公式: F(x) = sign(x) * log(1 + μ|x|) / log(1 + μ)
 * 然后量化到 [0, quantization_channels - 1]
 *
 * @param x - 输入信号，范围 [-1, 1]
 * @param quantization_channels - 量化级数 (默认: 256, 即 8-bit)
 * @returns 编码后的信号，整数值范围 [0, quantization_channels - 1]
 *
 * @example
 * ```ts
 * const signal = tensor([-0.5, 0, 0.5], { dtype: 'float32' });
 * const encoded = muLawEncoding(signal, 256);
 * // encoded 包含 [0, 256) 范围的整数
 * ```
 */
export function muLawEncoding(x: Tensor, quantization_channels: number = 256): Tensor {
    const mu = quantization_channels - 1;

    // F(x) = sign(x) * log(1 + μ|x|) / log(1 + μ)
    const absX = k.abs(x);
    const signX = k.sign(x);

    // log(1 + μ|x|) / log(1 + μ)
    const muTensor = k.mul(absX, mu);
    const numerator = k.log(k.add(muTensor, 1));
    const denominator = Math.log(1 + mu);

    const compressed = k.mul(signX, k.div(numerator, denominator));

    // 量化: 将 [-1, 1] 映射到 [0, quantizationChannels - 1]
    // (compressed + 1) / 2 * mu
    const shifted = k.add(compressed, 1);
    const scaled = k.mul(shifted, 0.5 * mu);

    // 四舍五入到整数
    const quantized = k.round(scaled);

    // 确保在有效范围内
    return k.clamp(quantized, 0, mu);
}

/**
 * μ-law 解码
 *
 * 将 [0, quantization_channels - 1] 范围的编码信号解码回 [-1, 1]
 *
 * 逆公式: x = sign(y) * (1/μ) * ((1 + μ)^|y| - 1)
 * 其中 y 是归一化到 [-1, 1] 的输入
 *
 * @param x_mu - 编码信号，整数值范围 [0, quantization_channels - 1]
 * @param quantization_channels - 量化级数 (默认: 256)
 * @returns 解码后的信号，范围 [-1, 1]
 *
 * @example
 * ```ts
 * const encoded = tensor([64, 128, 192], { dtype: 'float32' });
 * const decoded = muLawDecoding(encoded, 256);
 * // decoded 包含 [-1, 1] 范围的浮点数
 * ```
 */
export function muLawDecoding(x_mu: Tensor, quantization_channels: number = 256): Tensor {
    const mu = quantization_channels - 1;

    // 归一化到 [-1, 1]
    // y = 2 * x_mu / mu - 1
    const normalized = k.sub(k.mul(x_mu, 2 / mu), 1);

    // sign(y)
    const signY = k.sign(normalized);

    // |y|
    const absY = k.abs(normalized);

    // (1 + μ)^|y| - 1 = exp(|y| * log(1 + μ)) - 1
    const base = 1 + mu;
    const logBase = Math.log(base);
    const powered = k.sub(k.exp(k.mul(absY, logBase)), 1);

    // sign(y) * (1/μ) * ((1 + μ)^|y| - 1)
    const decoded = k.mul(signY, k.div(powered, mu));

    return decoded;
}
