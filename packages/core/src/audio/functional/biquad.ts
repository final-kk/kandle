/**
 * Biquad 滤波器系列
 *
 * 对标 torchaudio.functional 中的 biquad 系列函数
 *
 * biquad 是二阶 IIR 滤波器，系数计算公式参考:
 * https://www.w3.org/2011/audio/audio-eq-cookbook.html
 */

import { Tensor } from '../../tensor';
import { lfilterBiquad, type LfilterOptions } from './lfilter';

/**
 * Biquad 滤波器
 *
 * 执行二阶 IIR 滤波，初始条件设为 0。
 * 使用 GPU IIR kernel 进行优化计算。
 *
 * 传递函数:
 * ```
 * H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)
 * ```
 *
 * @param waveform - 输入波形 (..., time)
 * @param b0 - 分子系数 x[n]
 * @param b1 - 分子系数 x[n-1]
 * @param b2 - 分子系数 x[n-2]
 * @param a0 - 分母系数 y[n]，通常为 1
 * @param a1 - 分母系数 y[n-1]
 * @param a2 - 分母系数 y[n-2]
 * @param options - 可选参数
 * @returns 滤波后的波形
 *
 * @see https://en.wikipedia.org/wiki/Digital_biquad_filter
 */
export function biquad(
    waveform: Tensor,
    b0: number,
    b1: number,
    b2: number,
    a0: number,
    a1: number,
    a2: number,
    options?: LfilterOptions
): Tensor {
    // 归一化系数 (除以 a0)
    const b0Norm = b0 / a0;
    const b1Norm = b1 / a0;
    const b2Norm = b2 / a0;
    const a1Norm = a1 / a0;
    const a2Norm = a2 / a0;

    // 使用 IIR kernel (接收归一化的标量系数)
    return lfilterBiquad(waveform, b0Norm, b1Norm, b2Norm, a1Norm, a2Norm, options);
}

/**
 * 低通 Biquad 滤波器
 *
 * 设计并应用二阶低通滤波器，类似 SoX 实现。
 *
 * @param waveform - 输入波形 (..., time)
 * @param sample_rate - 采样率 (Hz)
 * @param cutoff_freq - 截止频率 (Hz)
 * @param Q - 品质因数，默认 0.707 (Butterworth)
 * @param options - 可选参数
 * @returns 滤波后的波形
 *
 * @see https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF
 */
export function lowpassBiquad(
    waveform: Tensor,
    sample_rate: number,
    cutoff_freq: number,
    Q: number = 0.707,
    options?: LfilterOptions
): Tensor {
    const w0 = (2 * Math.PI * cutoff_freq) / sample_rate;
    const alpha = Math.sin(w0) / (2 * Q);

    const b0 = (1 - Math.cos(w0)) / 2;
    const b1 = 1 - Math.cos(w0);
    const b2 = (1 - Math.cos(w0)) / 2;
    const a0 = 1 + alpha;
    const a1 = -2 * Math.cos(w0);
    const a2 = 1 - alpha;

    return biquad(waveform, b0, b1, b2, a0, a1, a2, options);
}

/**
 * 高通 Biquad 滤波器
 *
 * 设计并应用二阶高通滤波器，类似 SoX 实现。
 *
 * @param waveform - 输入波形 (..., time)
 * @param sample_rate - 采样率 (Hz)
 * @param cutoff_freq - 截止频率 (Hz)
 * @param Q - 品质因数，默认 0.707 (Butterworth)
 * @param options - 可选参数
 * @returns 滤波后的波形
 */
export function highpassBiquad(
    waveform: Tensor,
    sample_rate: number,
    cutoff_freq: number,
    Q: number = 0.707,
    options?: LfilterOptions
): Tensor {
    const w0 = (2 * Math.PI * cutoff_freq) / sample_rate;
    const alpha = Math.sin(w0) / (2 * Q);

    const b0 = (1 + Math.cos(w0)) / 2;
    const b1 = -(1 + Math.cos(w0));
    const b2 = (1 + Math.cos(w0)) / 2;
    const a0 = 1 + alpha;
    const a1 = -2 * Math.cos(w0);
    const a2 = 1 - alpha;

    return biquad(waveform, b0, b1, b2, a0, a1, a2, options);
}

/**
 * 带通 Biquad 滤波器
 *
 * 设计并应用二阶带通滤波器，类似 SoX 实现。
 *
 * @param waveform - 输入波形 (..., time)
 * @param sample_rate - 采样率 (Hz)
 * @param center_freq - 中心频率 (Hz)
 * @param Q - 品质因数，默认 0.707
 * @param const_skirt_gain - 如果为 true，使用恒定裙边增益 (峰值增益 = Q)
 * @param options - 可选参数
 * @returns 滤波后的波形
 */
export function bandpassBiquad(
    waveform: Tensor,
    sample_rate: number,
    center_freq: number,
    Q: number = 0.707,
    const_skirt_gain: boolean = false,
    options?: LfilterOptions
): Tensor {
    const w0 = (2 * Math.PI * center_freq) / sample_rate;
    const alpha = Math.sin(w0) / (2 * Q);

    const temp = const_skirt_gain ? Math.sin(w0) / 2 : alpha;
    const b0 = temp;
    const b1 = 0;
    const b2 = -temp;
    const a0 = 1 + alpha;
    const a1 = -2 * Math.cos(w0);
    const a2 = 1 - alpha;

    return biquad(waveform, b0, b1, b2, a0, a1, a2, options);
}

/**
 * 带阻 (陷波) Biquad 滤波器
 *
 * 设计并应用二阶带阻滤波器，类似 SoX 实现。
 *
 * @param waveform - 输入波形 (..., time)
 * @param sample_rate - 采样率 (Hz)
 * @param center_freq - 中心频率 (Hz)
 * @param Q - 品质因数，默认 0.707
 * @param options - 可选参数
 * @returns 滤波后的波形
 */
export function bandrejectBiquad(
    waveform: Tensor,
    sample_rate: number,
    center_freq: number,
    Q: number = 0.707,
    options?: LfilterOptions
): Tensor {
    const w0 = (2 * Math.PI * center_freq) / sample_rate;
    const alpha = Math.sin(w0) / (2 * Q);

    const b0 = 1;
    const b1 = -2 * Math.cos(w0);
    const b2 = 1;
    const a0 = 1 + alpha;
    const a1 = -2 * Math.cos(w0);
    const a2 = 1 - alpha;

    return biquad(waveform, b0, b1, b2, a0, a1, a2, options);
}

/**
 * 全通 Biquad 滤波器
 *
 * 设计并应用二阶全通滤波器，类似 SoX 实现。
 * 全通滤波器通过所有频率，但改变相位。
 *
 * @param waveform - 输入波形 (..., time)
 * @param sample_rate - 采样率 (Hz)
 * @param center_freq - 中心频率 (Hz)
 * @param Q - 品质因数，默认 0.707
 * @param options - 可选参数
 * @returns 滤波后的波形
 */
export function allpassBiquad(
    waveform: Tensor,
    sample_rate: number,
    center_freq: number,
    Q: number = 0.707,
    options?: LfilterOptions
): Tensor {
    const w0 = (2 * Math.PI * center_freq) / sample_rate;
    const alpha = Math.sin(w0) / (2 * Q);

    const b0 = 1 - alpha;
    const b1 = -2 * Math.cos(w0);
    const b2 = 1 + alpha;
    const a0 = 1 + alpha;
    const a1 = -2 * Math.cos(w0);
    const a2 = 1 - alpha;

    return biquad(waveform, b0, b1, b2, a0, a1, a2, options);
}

/**
 * 低频 (Bass) 音调控制滤波器
 *
 * 设计并应用低频增强/衰减滤波器，类似 SoX bass 效果。
 *
 * @param waveform - 输入波形 (..., time)
 * @param sample_rate - 采样率 (Hz)
 * @param gain - 增益量 (dB)，正值增强，负值衰减
 * @param center_freq - 中心频率 (Hz)，默认 100
 * @param Q - 品质因数，默认 0.707
 * @param options - 可选参数
 * @returns 滤波后的波形
 */
export function bassBiquad(
    waveform: Tensor,
    sample_rate: number,
    gain: number,
    center_freq: number = 100,
    Q: number = 0.707,
    options?: LfilterOptions
): Tensor {
    const w0 = (2 * Math.PI * center_freq) / sample_rate;
    const alpha = Math.sin(w0) / (2 * Q);
    const A = Math.exp((gain / 40) * Math.log(10));

    const temp1 = 2 * Math.sqrt(A) * alpha;
    const temp2 = (A - 1) * Math.cos(w0);
    const temp3 = (A + 1) * Math.cos(w0);

    const b0 = A * ((A + 1) - temp2 + temp1);
    const b1 = 2 * A * ((A - 1) - temp3);
    const b2 = A * ((A + 1) - temp2 - temp1);
    const a0 = (A + 1) + temp2 + temp1;
    const a1 = -2 * ((A - 1) + temp3);
    const a2 = (A + 1) + temp2 - temp1;

    return biquad(waveform, b0, b1, b2, a0, a1, a2, options);
}

/**
 * 高频 (Treble) 音调控制滤波器
 *
 * 设计并应用高频增强/衰减滤波器，类似 SoX treble 效果。
 *
 * @param waveform - 输入波形 (..., time)
 * @param sample_rate - 采样率 (Hz)
 * @param gain - 增益量 (dB)，正值增强，负值衰减
 * @param center_freq - 中心频率 (Hz)，默认 3000
 * @param Q - 品质因数，默认 0.707
 * @param options - 可选参数
 * @returns 滤波后的波形
 */
export function trebleBiquad(
    waveform: Tensor,
    sample_rate: number,
    gain: number,
    center_freq: number = 3000,
    Q: number = 0.707,
    options?: LfilterOptions
): Tensor {
    const w0 = (2 * Math.PI * center_freq) / sample_rate;
    const alpha = Math.sin(w0) / (2 * Q);
    const A = Math.exp((gain / 40) * Math.log(10));

    const temp1 = 2 * Math.sqrt(A) * alpha;
    const temp2 = (A - 1) * Math.cos(w0);
    const temp3 = (A + 1) * Math.cos(w0);

    const b0 = A * ((A + 1) + temp2 + temp1);
    const b1 = -2 * A * ((A - 1) + temp3);
    const b2 = A * ((A + 1) + temp2 - temp1);
    const a0 = (A + 1) - temp2 + temp1;
    const a1 = 2 * ((A - 1) - temp3);
    const a2 = (A + 1) - temp2 - temp1;

    return biquad(waveform, b0, b1, b2, a0, a1, a2, options);
}

/**
 * 峰值均衡器 Biquad 滤波器
 *
 * 设计并应用峰值均衡器滤波器，类似 SoX equalizer 效果。
 *
 * @param waveform - 输入波形 (..., time)
 * @param sample_rate - 采样率 (Hz)
 * @param center_freq - 中心频率 (Hz)
 * @param gain - 增益量 (dB)
 * @param Q - 品质因数，默认 0.707
 * @param options - 可选参数
 * @returns 滤波后的波形
 */
export function equalizerBiquad(
    waveform: Tensor,
    sample_rate: number,
    center_freq: number,
    gain: number,
    Q: number = 0.707,
    options?: LfilterOptions
): Tensor {
    const w0 = (2 * Math.PI * center_freq) / sample_rate;
    const A = Math.exp((gain / 40) * Math.log(10));
    const alpha = Math.sin(w0) / (2 * Q);

    const b0 = 1 + alpha * A;
    const b1 = -2 * Math.cos(w0);
    const b2 = 1 - alpha * A;
    const a0 = 1 + alpha / A;
    const a1 = -2 * Math.cos(w0);
    const a2 = 1 - alpha / A;

    return biquad(waveform, b0, b1, b2, a0, a1, a2, options);
}
