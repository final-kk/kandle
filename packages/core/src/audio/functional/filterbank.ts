/**
 * Audio Filterbank Functions
 *
 * 对标 torchaudio.functional 中的滤波器组函数：
 * - melscale_fbanks: Mel 滤波器组矩阵
 * - linear_fbanks: 线性三角滤波器组
 * - create_dct: DCT 变换矩阵
 */

import { Tensor } from '../../tensor';

// ============================================================================
// Mel 频率转换辅助函数
// ============================================================================

/**
 * Hz 到 Mel 频率转换 (HTK 公式)
 */
function hzToMelHTK(hz: number): number {
    return 2595.0 * Math.log10(1.0 + hz / 700.0);
}

/**
 * Mel 到 Hz 频率转换 (HTK 公式)
 */
function melToHzHTK(mel: number): number {
    return 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0);
}

/**
 * Hz 到 Mel 频率转换 (Slaney 公式)
 * Slaney 使用线性+对数混合
 */
function hzToMelSlaney(hz: number): number {
    const f_min = 0.0;
    const f_sp = 200.0 / 3.0; // ~66.67 Hz
    const min_log_hz = 1000.0;
    const min_log_mel = (min_log_hz - f_min) / f_sp;
    const logstep = Math.log(6.4) / 27.0;

    if (hz >= min_log_hz) {
        return min_log_mel + Math.log(hz / min_log_hz) / logstep;
    }
    return (hz - f_min) / f_sp;
}

/**
 * Mel 到 Hz 频率转换 (Slaney 公式)
 */
function melToHzSlaney(mel: number): number {
    const f_min = 0.0;
    const f_sp = 200.0 / 3.0;
    const min_log_hz = 1000.0;
    const min_log_mel = (min_log_hz - f_min) / f_sp;
    const logstep = Math.log(6.4) / 27.0;

    if (mel >= min_log_mel) {
        return min_log_hz * Math.exp(logstep * (mel - min_log_mel));
    }
    return f_min + f_sp * mel;
}

// ============================================================================
// 公开 API
// ============================================================================

export type MelScaleType = 'htk' | 'slaney';
export type NormType = 'slaney' | null;

/**
 * 创建 Mel 滤波器组矩阵
 *
 * @param n_freqs - STFT 输出的频率 bins 数量 (通常为 n_fft // 2 + 1)
 * @param f_min - 最小频率 (Hz)
 * @param f_max - 最大频率 (Hz)
 * @param n_mels - Mel 滤波器数量
 * @param sample_rate - 采样率 (Hz)
 * @param norm - 归一化方式: 'slaney' 按带宽归一化，null 不归一化
 * @param mel_scale - Mel 频率转换公式: 'htk' 或 'slaney'
 * @returns 形状为 [n_freqs, n_mels] 的滤波器组矩阵
 *
 * @example
 * ```ts
 * const fbanks = melscaleFbanks(201, 0, 8000, 80, 16000, 'slaney', 'htk');
 * // fbanks.shape = [201, 80]
 * ```
 */
export function melscaleFbanks(
    n_freqs: number,
    f_min: number,
    f_max: number,
    n_mels: number,
    sample_rate: number,
    norm: NormType = null,
    mel_scale: MelScaleType = 'htk'
): Tensor {
    // 选择转换函数
    const hzToMel = mel_scale === 'htk' ? hzToMelHTK : hzToMelSlaney;
    const melToHz = mel_scale === 'htk' ? melToHzHTK : melToHzSlaney;

    // 计算 Mel 频率边界点 (n_mels + 2 个点)
    const melMin = hzToMel(f_min);
    const melMax = hzToMel(f_max);
    const melPoints = new Float32Array(n_mels + 2);
    for (let i = 0; i < n_mels + 2; i++) {
        const mel = melMin + (melMax - melMin) * i / (n_mels + 1);
        melPoints[i] = melToHz(mel);
    }

    // 计算 FFT bin 频率
    const fftFreqs = new Float32Array(n_freqs);
    for (let i = 0; i < n_freqs; i++) {
        fftFreqs[i] = (sample_rate / 2) * i / (n_freqs - 1);
    }

    // 构建滤波器组矩阵
    const fbanks = new Float32Array(n_freqs * n_mels);

    for (let m = 0; m < n_mels; m++) {
        const f_left = melPoints[m];
        const f_center = melPoints[m + 1];
        const f_right = melPoints[m + 2];

        for (let k = 0; k < n_freqs; k++) {
            const f = fftFreqs[k];

            if (f >= f_left && f <= f_center) {
                // 上升斜坡
                fbanks[k * n_mels + m] = (f - f_left) / (f_center - f_left);
            } else if (f > f_center && f <= f_right) {
                // 下降斜坡
                fbanks[k * n_mels + m] = (f_right - f) / (f_right - f_center);
            }
            // 其他情况保持为 0
        }

        // Slaney 归一化: 除以带宽
        if (norm === 'slaney') {
            const bandwidth = f_right - f_left;
            if (bandwidth > 0) {
                const enorm = 2.0 / bandwidth;
                for (let k = 0; k < n_freqs; k++) {
                    fbanks[k * n_mels + m] *= enorm;
                }
            }
        }
    }

    return new Tensor(fbanks, { shape: [n_freqs, n_mels], dtype: 'float32' });
}

/**
 * 创建线性三角滤波器组 (用于 LFCC)
 *
 * @param n_freqs - STFT 输出的频率 bins 数量
 * @param f_min - 最小频率 (Hz)
 * @param f_max - 最大频率 (Hz)
 * @param n_filter - 滤波器数量
 * @param sample_rate - 采样率 (Hz)
 * @returns 形状为 [n_freqs, n_filter] 的滤波器组矩阵
 */
export function linearFbanks(
    n_freqs: number,
    f_min: number,
    f_max: number,
    n_filter: number,
    sample_rate: number
): Tensor {
    // 线性分布的频率边界点
    const freqPoints = new Float32Array(n_filter + 2);
    for (let i = 0; i < n_filter + 2; i++) {
        freqPoints[i] = f_min + (f_max - f_min) * i / (n_filter + 1);
    }

    // FFT bin 频率
    const fftFreqs = new Float32Array(n_freqs);
    for (let i = 0; i < n_freqs; i++) {
        fftFreqs[i] = (sample_rate / 2) * i / (n_freqs - 1);
    }

    // 构建滤波器组
    const fbanks = new Float32Array(n_freqs * n_filter);

    for (let m = 0; m < n_filter; m++) {
        const f_left = freqPoints[m];
        const f_center = freqPoints[m + 1];
        const f_right = freqPoints[m + 2];

        for (let k = 0; k < n_freqs; k++) {
            const f = fftFreqs[k];

            if (f >= f_left && f <= f_center) {
                fbanks[k * n_filter + m] = (f - f_left) / (f_center - f_left);
            } else if (f > f_center && f <= f_right) {
                fbanks[k * n_filter + m] = (f_right - f) / (f_right - f_center);
            }
        }
    }

    return new Tensor(fbanks, { shape: [n_freqs, n_filter], dtype: 'float32' });
}

/**
 * 创建 DCT (离散余弦变换) 矩阵
 *
 * 用于 MFCC/LFCC 计算，将 Mel/线性频谱转换为倒谱系数
 *
 * @param n_mfcc - 输出 MFCC 系数数量
 * @param n_mels - 输入 Mel 频带数量
 * @param norm - 归一化方式: 'ortho' 正交归一化，null 不归一化
 * @returns 形状为 [n_mels, n_mfcc] 的 DCT 矩阵
 *
 * @example
 * ```ts
 * const dctMatrix = createDct(13, 80, 'ortho');
 * // mfcc = melSpec @ dctMatrix
 * ```
 */
export function createDct(
    n_mfcc: number,
    n_mels: number,
    norm: 'ortho' | null = 'ortho'
): Tensor {
    // DCT-II 公式: dct[n, k] = cos(π * k * (n + 0.5) / N)
    // 其中 n 是输入索引 [0, n_mels), k 是输出索引 [0, n_mfcc)

    const dct = new Float32Array(n_mels * n_mfcc);

    for (let n = 0; n < n_mels; n++) {
        for (let k = 0; k < n_mfcc; k++) {
            dct[n * n_mfcc + k] = Math.cos(Math.PI * k * (n + 0.5) / n_mels);
        }
    }

    // 正交归一化
    if (norm === 'ortho') {
        const scale0 = Math.sqrt(1.0 / n_mels);
        const scaleK = Math.sqrt(2.0 / n_mels);

        for (let n = 0; n < n_mels; n++) {
            // k=0 列使用 scale0
            dct[n * n_mfcc + 0] *= scale0;
            // 其他列使用 scaleK
            for (let k = 1; k < n_mfcc; k++) {
                dct[n * n_mfcc + k] *= scaleK;
            }
        }
    }

    return new Tensor(dct, { shape: [n_mels, n_mfcc], dtype: 'float32' });
}
