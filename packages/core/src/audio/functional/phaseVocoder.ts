/**
 * Phase Vocoder
 *
 * 对标 torchaudio.functional.phase_vocoder
 *
 * 实现时间拉伸而不改变音高的核心算法
 */

import { type Tensor } from '../../tensor';
import * as k from '../../index';

/**
 * 从幅度和相位创建复数张量 (polar form)
 *
 * z = r * (cos(θ) + i*sin(θ)) = r * e^(iθ)
 *
 * @param abs - 幅度张量 r
 * @param angle - 相位张量 θ (弧度)
 * @returns 复数张量
 */
function polar(abs: Tensor, angle: Tensor): Tensor {
    const realPart = k.mul(abs, k.cos(angle)) as Tensor<'float32'>;
    const imagPart = k.mul(abs, k.sin(angle)) as Tensor<'float32'>;
    return k.torchComplex(realPart, imagPart);
}

/**
 * 生成沿最后一维的切片表达式
 * 对于 shape [a, b, c]，生成 ":,:,start:end"
 * 对于 shape [a, b]，生成 ":,start:end"
 */
function sliceLastDim(ndim: number, start: number, end: number): string {
    const prefix = ':,'.repeat(ndim - 1);
    return `${prefix}${start}:${end}`;
}

/**
 * 通过相位声码器实现时间拉伸
 *
 * Phase Vocoder 通过在频域中操作 STFT 来改变音频的时间尺度而不影响音高。
 * 算法通过维护累积相位并使用瞬时频率来重建正确的相位关系。
 *
 * @param complex_specgrams - 复数 STFT 频谱图，形状 (..., freq, time)
 * @param rate - 拉伸因子。rate > 1 加速 (缩短)，rate < 1 减速 (延长)
 * @param phase_advance - 每个频率 bin 的期望相位增量，形状 (freq,) 或 (freq, 1)
 *                       通常为 `linspace(0, pi * hop_length, n_fft // 2 + 1)`
 * @returns 时间拉伸后的复数频谱图，形状 (..., freq, ceil(time / rate))
 *
 * @example
 * ```ts
 * // 创建 phase_advance
 * const n_fft = 512;
 * const hop_length = 128;
 * const phase_advance = k.linspace(0, Math.PI * hop_length, n_fft / 2 + 1);
 *
 * // 时间拉伸 (减速 50%)
 * const stretched = phaseVocoder(stftResult, 0.5, phase_advance);
 * ```
 */
export function phaseVocoder(
    complex_specgrams: Tensor,
    rate: number,
    phase_advance: Tensor
): Tensor {
    // 验证输入
    if (!complex_specgrams.dtype.startsWith('complex')) {
        throw new Error(
            `phase_vocoder: input must be complex, got ${complex_specgrams.dtype}`
        );
    }

    if (rate <= 0) {
        throw new Error(`phase_vocoder: rate must be positive, got ${rate}`);
    }

    const shape = complex_specgrams.shape;
    const ndim = shape.length;
    if (ndim < 2) {
        throw new Error(
            `phase_vocoder: input must have at least 2 dimensions, got ${ndim}`
        );
    }

    const numFreqs = shape[ndim - 2];
    const numFrames = shape[ndim - 1];

    // 确保 phase_advance 的形状正确 (freq, 1)
    let phaseAdv = phase_advance;
    if (phaseAdv.shape.length === 1) {
        phaseAdv = k.unsqueeze(phaseAdv, -1);
    }

    // 计算输出帧数
    const numOutputFrames = Math.ceil(numFrames / rate);

    // 获取输入的幅度和相位
    const inputMag = k.abs(complex_specgrams);
    const inputPhase = k.angle(complex_specgrams);

    // 预计算相位差分 (instantaneous frequency deviation)
    // phaseDiff[..., t] = angle(spec[..., t+1]) - angle(spec[..., t]) - phaseAdvance
    // wrap 到 [-pi, pi]，然后加回 phaseAdvance
    const phase0 = k.slice(inputPhase, sliceLastDim(ndim, 0, numFrames - 1));
    const phase1 = k.slice(inputPhase, sliceLastDim(ndim, 1, numFrames));
    let phaseDiff = k.sub(phase1, phase0);
    phaseDiff = k.sub(phaseDiff, phaseAdv);
    // Wrap 到 [-pi, pi] 使用 remainder
    // x = (x + pi) % (2pi) - pi
    // 使用 k.remainder (scalar)
    const twoPi = 2 * Math.PI;
    phaseDiff = k.add(phaseDiff, Math.PI);
    phaseDiff = k.remainder(phaseDiff, twoPi);
    phaseDiff = k.sub(phaseDiff, Math.PI);

    // 加回期望增量
    phaseDiff = k.add(phaseDiff, phaseAdv);

    // 处理输出帧
    const outputFrames: Tensor[] = [];

    // 第一帧: 使用输入的第一帧
    const mag0 = k.slice(inputMag, sliceLastDim(ndim, 0, 1));
    const phase0Frame = k.slice(inputPhase, sliceLastDim(ndim, 0, 1));
    outputFrames.push(polar(mag0, phase0Frame));

    let cumulativePhase = phase0Frame;

    // 处理后续帧
    for (let i = 1; i < numOutputFrames; i++) {
        const t = i * rate;
        const t0 = Math.floor(t);
        const t1 = Math.min(t0 + 1, numFrames - 1);
        const alpha = t - t0;

        // 线性插值幅度
        const mag_t0 = k.slice(inputMag, sliceLastDim(ndim, t0, t0 + 1));
        const mag_t1 = k.slice(inputMag, sliceLastDim(ndim, t1, t1 + 1));
        const interpMag = k.add(
            k.mul(mag_t0, 1 - alpha),
            k.mul(mag_t1, alpha)
        );

        // 累积相位更新
        // 使用 floor(t) 对应的 phaseDiff，乘以 rate
        const diffIdx = Math.min(t0, numFrames - 2);
        const diff = k.slice(phaseDiff, sliceLastDim(ndim, diffIdx, diffIdx + 1));
        cumulativePhase = k.add(cumulativePhase, k.mul(diff, rate));

        // 构建复数输出帧
        outputFrames.push(polar(interpMag, cumulativePhase));
    }

    // 沿时间轴拼接所有帧
    return k.cat(outputFrames, -1);
}
