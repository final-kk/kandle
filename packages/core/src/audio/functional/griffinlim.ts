/**
 * Griffin-Lim Algorithm
 *
 * 对标 torchaudio.functional.griffinlim
 *
 * 从幅度频谱图重建波形 (相位恢复)
 */

import { type Tensor } from '../../tensor';
import * as k from '../../index';

/**
 * 从幅度和相位创建复数张量 (polar form)
 */
function polar(abs: Tensor, angle: Tensor): Tensor {
    const realPart = k.mul(abs, k.cos(angle)) as Tensor<'float32'>;
    const imagPart = k.mul(abs, k.sin(angle)) as Tensor<'float32'>;
    return k.torchComplex(realPart, imagPart);
}

/**
 * 使用 Griffin-Lim 算法从幅度频谱图重建波形
 *
 * Griffin-Lim 是一种迭代算法，通过交替进行 STFT 和 iSTFT
 * 来恢复与给定幅度谱图一致的相位信息。
 *
 * @param specgram - 幅度频谱图 (..., freq, time)，其中 freq = n_fft // 2 + 1
 * @param window - 窗函数张量
 * @param n_fft - FFT 大小
 * @param hop_length - 帧移
 * @param win_length - 窗长度
 * @param power - 输入频谱图的功率指数 (1=幅度, 2=功率)
 * @param n_iter - 迭代次数 (通常 32)
 * @param momentum - 快速 Griffin-Lim 动量 (0=标准, 0.99=快速)
 * @param length - 输出波形长度 (可选)
 * @param rand_init - 是否随机初始化相位 (默认 true)
 * @returns 重建的波形 (..., time)
 *
 * @example
 * ```ts
 * // 从幅度谱重建波形
 * const window = k.hannWindow(n_fft);
 * const magSpec = k.abs(k.stft(waveform, n_fft));
 * const reconstructed = griffinlim(
 *     magSpec, window, n_fft, hop_length, n_fft, 1, 32, 0.99
 * );
 * ```
 */
export function griffinlim(
    specgram: Tensor,
    window: Tensor,
    n_fft: number,
    hop_length: number,
    win_length: number,
    power: number,
    n_iter: number,
    momentum: number,
    length?: number,
    rand_init: boolean = true
): Tensor {
    // 验证输入
    if (specgram.dtype.startsWith('complex')) {
        throw new Error(
            `griffinlim: input must be magnitude (real), got ${specgram.dtype}`
        );
    }

    if (power <= 0) {
        throw new Error(`griffinlim: power must be positive, got ${power}`);
    }

    if (n_iter < 1) {
        throw new Error(`griffinlim: n_iter must be >= 1, got ${n_iter}`);
    }

    if (momentum < 0) {
        throw new Error(`griffinlim: momentum must be >= 0, got ${momentum}`);
    }

    const shape = specgram.shape;
    const ndim = shape.length;
    const numFreqs = shape[ndim - 2];
    const numFrames = shape[ndim - 1];

    // 如果输入是功率谱，转换为幅度谱
    let magnitudes: Tensor;
    if (power === 2) {
        magnitudes = k.sqrt(specgram);
    } else if (power === 1) {
        magnitudes = specgram;
    } else {
        // power 次方根: x^(1/power)
        magnitudes = k.pow(specgram, 1 / power);
    }

    // 初始化相位
    let angles: Tensor;
    if (rand_init) {
        // 随机相位 [-pi, pi]
        const randUniform = k.rand([...shape], specgram.dtype);
        angles = k.sub(k.mul(randUniform, 2 * Math.PI), Math.PI);
    } else {
        // 零相位
        angles = k.zeros([...shape], specgram.dtype);
    }

    // 构建初始复数频谱图
    let rebuilt = polar(magnitudes, angles);

    // 快速 Griffin-Lim: 保存前一次重建的频谱图
    let tprev = rebuilt;

    // 迭代
    for (let i = 0; i < n_iter; i++) {
        // Inverse STFT: 频域 -> 时域
        const waveform = k.istft(
            rebuilt,
            n_fft,
            hop_length,
            win_length,
            window,
            true,  // center
            false, // normalized
            true,  // onesided
            length
        );

        // Forward STFT: 时域 -> 频域
        rebuilt = k.stft(
            waveform,
            n_fft,
            hop_length,
            win_length,
            window,
            true,  // center
            'reflect', // pad_mode
            false, // normalized
            true,  // onesided
            true   // return_complex
        );

        // 提取新相位
        angles = k.angle(rebuilt);

        // 快速 Griffin-Lim: 应用动量
        if (momentum > 0 && i > 0) {
            // angle_adjustment = angle(rebuilt) - angle(tprev)
            const tprevAngles = k.angle(tprev);
            const angleUpdate = k.sub(angles, tprevAngles);
            angles = k.sub(angles, k.mul(angleUpdate, momentum));
        }

        // 保存当前重建
        tprev = rebuilt;

        // 用目标幅度替换重建幅度，保留相位
        rebuilt = polar(magnitudes, angles);
    }

    // 最终 iSTFT 输出波形
    return k.istft(
        rebuilt,
        n_fft,
        hop_length,
        win_length,
        window,
        true,  // center
        false, // normalized
        true,  // onesided
        length
    );
}
