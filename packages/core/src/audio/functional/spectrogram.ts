/**
 * Spectrogram Functions
 *
 * 对标 torchaudio.functional.spectrogram 和 inverse_spectrogram
 */

import { type Tensor } from '../../tensor';
import * as k from '../../index';

/**
 * 从音频波形创建频谱图
 *
 * 这是对 STFT 的高级封装，支持指定 power 参数
 *
 * @param waveform - 输入波形张量 (..., time)
 * @param pad - 两侧 padding 的采样点数
 * @param window - 窗函数张量
 * @param n_fft - FFT 大小
 * @param hop_length - 帧移
 * @param win_length - 窗长度
 * @param power - 功率指数: null=复数, 1=幅度, 2=功率
 * @param normalized - 是否归一化 STFT
 * @param center - 是否中心 padding (在 stft 内部处理)
 * @param pad_mode - padding 模式
 * @param onesided - 是否只返回正频率
 * @returns 频谱图张量 (..., n_freqs, time)
 */
export function spectrogram(
    waveform: Tensor,
    pad: number,
    window: Tensor,
    n_fft: number,
    hop_length: number,
    win_length: number,
    power: number | null = null,
    normalized: boolean = false,
    center: boolean = true,
    pad_mode: 'constant' | 'reflect' | 'replicate' | 'circular' = 'reflect',
    onesided: boolean = true
): Tensor {
    // Step 1: 前置 padding (如果 pad > 0)
    let signal = waveform;
    if (pad > 0) {
        signal = k.pad(signal, [pad, pad], 'constant', 0);
    }

    // Step 2: STFT
    const stftResult = k.stft(
        signal,
        n_fft,
        hop_length,
        win_length,
        window,
        center,
        pad_mode,
        normalized,
        onesided,
        true // return_complex
    );

    // Step 3: 根据 power 参数处理
    if (power === null) {
        // 返回复数 STFT
        return stftResult;
    } else if (power === 1) {
        // 返回幅度谱
        return k.abs(stftResult);
    } else if (power === 2) {
        // 返回功率谱
        const absSpec = k.abs(stftResult);
        return k.mul(absSpec, absSpec);
    } else {
        // 任意 power
        return k.pow(k.abs(stftResult), power);
    }
}

/**
 * 从复数频谱图重建波形
 *
 * @param spec - 复数频谱图张量 (..., n_freqs, time)
 * @param length - 输出波形长度 (可选)
 * @param pad - 原始 spectrogram 时使用的 pad 值
 * @param window - 窗函数张量
 * @param n_fft - FFT 大小
 * @param hop_length - 帧移
 * @param win_length - 窗长度
 * @param normalized - 是否归一化
 * @param center - 是否使用 center padding
 * @param onesided - 是否为单边频谱
 * @returns 重建的波形张量 (..., time)
 */
export function inverseSpectrogram(
    spec: Tensor,
    length?: number,
    pad: number = 0,
    window?: Tensor,
    n_fft?: number,
    hop_length?: number,
    win_length?: number,
    normalized: boolean = false,
    center: boolean = true,
    onesided: boolean = true
): Tensor {
    // 从频谱图形状推断 n_fft (如果未指定)
    const n_freqs = spec.shape[spec.shape.length - 2];
    const inferredNfft = onesided ? (n_freqs - 1) * 2 : n_freqs;
    const actualNfft = n_fft ?? inferredNfft;
    const actualHopLength = hop_length ?? Math.floor(actualNfft / 4);
    const actualWinLength = win_length ?? actualNfft;

    // 调用 istft
    let result = k.istft(
        spec,
        actualNfft,
        actualHopLength,
        actualWinLength,
        window,
        center,
        normalized,
        onesided,
        length
    );

    // 移除前置 padding (如果有)
    if (pad > 0) {
        const ndim = result.shape.length;
        const signalLen = result.shape[ndim - 1];
        // Slice: [..., pad:-pad]
        result = k.slice(result, `${pad}:${signalLen - pad}`);
    }

    return result;
}
