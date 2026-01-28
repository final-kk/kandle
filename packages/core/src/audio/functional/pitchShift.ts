/**
 * Pitch Shift
 *
 * 对标 torchaudio.functional.pitch_shift
 *
 * 使用 Phase Vocoder + Resample 实现音高变换
 */

import { type Tensor } from '../../tensor';
import * as k from '../../index';
import { phaseVocoder } from './phaseVocoder';
import { resample } from './resample';

export interface PitchShiftOptions {
    /** 每八度的音阶数 (默认: 12, 半音) */
    bins_per_octave?: number;
    /** FFT 大小 (默认: 512) */
    n_fft?: number;
    /** 窗口长度 (默认: n_fft) */
    win_length?: number;
    /** 帧移长度 (默认: win_length // 4) */
    hop_length?: number;
    /** 窗函数张量 (默认: hann) */
    window?: Tensor;
}

/**
 * 计算最大公约数
 */
function gcd(a: number, b: number): number {
    a = Math.abs(Math.floor(a));
    b = Math.abs(Math.floor(b));
    while (b !== 0) {
        const temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

/**
 * 将比例近似为具有较大 GCD 的分数
 * 
 * 例如: rate=1.2599 ≈ 5/4 (GCD with 16000 = 4000) 而不是 20159/16000 (GCD=1)
 */
function approximateRate(rate: number, maxDenominator: number = 64): { numerator: number; denominator: number } {
    // 使用连分数展开获得最佳有理近似
    let bestNum = 1;
    let bestDen = 1;
    let bestError = Math.abs(rate - 1);

    for (let den = 1; den <= maxDenominator; den++) {
        const num = Math.round(rate * den);
        if (num <= 0) continue;

        const error = Math.abs(rate - num / den);
        if (error < bestError) {
            bestError = error;
            bestNum = num;
            bestDen = den;
        }

        // 如果误差足够小就停止
        if (error < 0.001) break;
    }

    // 约简分数
    const g = gcd(bestNum, bestDen);
    return { numerator: bestNum / g, denominator: bestDen / g };
}

/**
 * 音高变换
 *
 * 在不改变时长的情况下改变音高。
 * 正数 n_steps 提高音高，负数降低音高。
 *
 * @param waveform - 输入波形 (..., time)
 * @param sample_rate - 采样率 (Hz)
 * @param n_steps - 音高变换步数 (半音制下 12 为一个八度)
 * @param options - 可选参数
 * @returns 变换后的波形 (..., time)
 *
 * @example
 * ```ts
 * // 升高一个八度 (12 个半音)
 * const higher = pitchShift(waveform, 16000, 12);
 *
 * // 降低一个半音
 * const lower = pitchShift(waveform, 16000, -1);
 * ```
 */
export function pitchShift(
    waveform: Tensor,
    sample_rate: number,
    n_steps: number,
    options: PitchShiftOptions = {}
): Tensor {
    // 如果 n_steps 为 0，直接返回
    if (n_steps === 0) {
        return waveform;
    }

    const bins_per_octave = options.bins_per_octave ?? 12;
    const n_fft = options.n_fft ?? 512;
    const win_length = options.win_length ?? n_fft;
    const hop_length = options.hop_length ?? Math.floor(win_length / 4);
    const window = options.window ?? k.hannWindow(win_length);

    // 计算时间拉伸率
    // rate = 2^(n_steps / bins_per_octave)
    const rate = Math.pow(2, n_steps / bins_per_octave);

    // 将比例近似为简单分数，避免 resample 产生巨大的 kernel
    const { numerator, denominator } = approximateRate(rate, 64);
    const approxRate = numerator / denominator;

    // 保存原始形状
    const origShape = waveform.shape;
    const origLength = origShape[origShape.length - 1];

    // 展平为 (batch, time)
    let flat = waveform;
    if (origShape.length === 1) {
        flat = k.unsqueeze(waveform, 0);
    } else if (origShape.length > 2) {
        // 合并前面的维度
        const batchSize = origShape.slice(0, -1).reduce((a, b) => a * b, 1);
        flat = k.reshape(waveform, [batchSize, origLength]);
    }

    // Step 1: STFT
    const stftResult = k.stft(
        flat,
        n_fft,
        hop_length,
        win_length,
        window,
        true,     // center
        'reflect', // pad_mode
        false,    // normalized
        true,     // onesided
        true      // return_complex
    );

    // Step 2: Phase Vocoder (时间拉伸)
    const numFreqs = n_fft / 2 + 1;
    // const step = (Math.PI * hop_length) / (numFreqs - 1);
    const phaseAdvance = k.linspace(0, Math.PI * hop_length, numFreqs, 'float32');

    // 使用近似的 rate 进行 phase vocoder
    const stretched = phaseVocoder(stftResult, approxRate, phaseAdvance);

    // Step 3: iSTFT (重建波形)
    const stretchedWaveform = k.istft(
        stretched,
        n_fft,
        hop_length,
        win_length,
        window,
        true,     // center
        false,    // normalized
        true,     // onesided
        undefined // length
    );

    // Step 4: Resample 恢复原始长度并改变音高
    // 使用简化的采样率比例: sample_rate * denominator -> sample_rate * numerator
    // 这样 gcd = sample_rate，大大减少 kernel 大小
    const origFreq = sample_rate * denominator;
    const newFreq = sample_rate * numerator;
    const resampled = resample(stretchedWaveform, origFreq, newFreq);

    // Step 5: 裁剪或填充到原始长度
    const resampledLength = resampled.shape[resampled.shape.length - 1];
    let result: Tensor;

    if (resampledLength === origLength) {
        result = resampled;
    } else if (resampledLength > origLength) {
        // 裁剪: 使用 Python 风格切片
        result = k.slice(resampled, `..., :${origLength}`);
    } else {
        // 填充 (零填充)
        const padAmount = origLength - resampledLength;
        result = k.pad(resampled, [0, padAmount], 'constant', 0);
    }

    // 恢复原始形状
    if (origShape.length === 1) {
        result = k.squeeze(result, 0);
    } else if (origShape.length > 2) {
        result = k.reshape(result, [...origShape] as number[]);
    }

    return result;
}
