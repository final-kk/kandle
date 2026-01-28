/**
 * Delta Coefficients
 *
 * 对标 torchaudio.functional.compute_deltas
 */

import { type Tensor } from '../../tensor';
import * as k from '../../index';

/**
 * 计算帧序列的 delta 系数
 *
 * Delta 系数表示特征的时间变化率，常用于语音识别中增强 MFCC 特征
 *
 * @param specgram - 输入特征张量 (..., n_features, time)
 * @param win_length - 差分窗口长度 (必须为奇数)
 * @param mode - padding 模式: 'replicate' 或 'reflect'
 * @returns delta 系数张量，形状与输入相同
 *
 * @example
 * ```ts
 * // 计算一阶 delta
 * const delta = computeDeltas(mfcc, 5);
 * // 计算二阶 delta (delta-delta)
 * const deltaDelta = computeDeltas(delta, 5);
 * ```
 */
export function computeDeltas(
    specgram: Tensor,
    win_length: number = 5,
    mode: 'replicate' | 'reflect' = 'replicate'
): Tensor {
    // 验证参数
    if (win_length < 3 || win_length % 2 === 0) {
        throw new Error(`compute_deltas: win_length must be odd and >= 3, got ${win_length}`);
    }

    const n = Math.floor(win_length / 2);

    // 计算归一化因子: sum(i^2 for i in -n to n) = 2 * sum(i^2 for i=1 to n)
    // = 2 * n * (n + 1) * (2n + 1) / 6 = n * (n + 1) * (2n + 1) / 3
    const normFactor = n * (n + 1) * (2 * n + 1) / 3;

    // Padding 沿时间维度 (最后一维)
    const padded = k.pad(specgram, [n, n], mode);

    // 获取形状信息
    const shape = specgram.shape;
    const ndim = shape.length;
    const timeLen = shape[ndim - 1];

    // 构建切片前缀（对于所有非时间维度）
    const slicePrefix = ':,'.repeat(ndim - 1);

    // 简化实现：使用切片和加权求和
    // delta = sum(i * (x[..., n+i:n+i+timeLen] - x[..., n-i:n-i+timeLen]) for i in 1 to n) / normFactor

    let delta = k.zeros([...shape], specgram.dtype);

    for (let i = 1; i <= n; i++) {
        const sliceStart1 = n + i;
        const sliceStart2 = n - i;

        // 构建完整的切片表达式: ":,...,:,start:start+timeLen"
        const sliceExpr1 = `${slicePrefix}${sliceStart1}:${sliceStart1 + timeLen}`;
        const sliceExpr2 = `${slicePrefix}${sliceStart2}:${sliceStart2 + timeLen}`;

        const part1 = k.slice(padded, sliceExpr1);
        const part2 = k.slice(padded, sliceExpr2);

        const diff = k.sub(part1, part2);
        const weighted = k.mul(diff, i / normFactor);
        delta = k.add(delta, weighted);
    }

    return delta;
}
