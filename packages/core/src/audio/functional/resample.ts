/**
 * Audio Resample Functions
 *
 * 对标 torchaudio.functional.resample
 * 
 * 使用带限 sinc 插值实现高质量重采样
 * 
 * @see https://pytorch.org/audio/stable/generated/torchaudio.functional.resample.html
 */

import { Tensor } from '../../tensor';
import * as k from '../../index';

/**
 * 计算最大公约数 (Euclidean算法)
 */
export function gcd(a: number, b: number): number {
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
 * 重采样选项
 */
export interface ResampleOptions {
    /**
     * 低通滤波器宽度 (零交叉次数)
     * 更大的值 = 更锐利的滤波器，但计算更慢
     * @default 6
     */
    lowpass_filter_width?: number;

    /**
     * 滤波器截止频率，作为 Nyquist 频率的比例
     * 较低的值减少混叠，但也会衰减高频
     * @default 0.99
     */
    rolloff?: number;

    /**
     * 重采样方法
     * - 'sinc_interp_hann': 使用 Hann 窗 (默认)
     * - 'sinc_interp_kaiser': 使用 Kaiser 窗
     * @default 'sinc_interp_hann'
     */
    resampling_method?: 'sinc_interp_hann' | 'sinc_interp_kaiser';

    /**
     * Kaiser 窗的形状参数
     * 仅在 resampling_method === 'sinc_interp_kaiser' 时使用
     * @default 14.769656459379492
     */
    beta?: number;
}

/**
 * 生成 sinc 重采样内核
 * 
 * 用于预计算重采样 kernel，供 Transform 类缓存使用
 */
export function getSincResampleKernel(
    origFreq: number,
    newFreq: number,
    gcdVal: number,
    lowpassFilterWidth: number,
    rolloff: number,
    resamplingMethod: 'sinc_interp_hann' | 'sinc_interp_kaiser',
    beta: number | undefined,
): { kernel: Tensor; width: number } {
    // 简化频率
    const origFreqReduced = Math.floor(origFreq / gcdVal);
    const newFreqReduced = Math.floor(newFreq / gcdVal);

    // 计算基础频率 (用于抗混叠)
    let baseFreq = Math.min(origFreqReduced, newFreqReduced);
    baseFreq *= rolloff;

    // 计算滤波器宽度
    const width = Math.ceil(lowpassFilterWidth * origFreqReduced / baseFreq);

    // 生成索引: [-width, width + origFreqReduced) / origFreqReduced
    const numIdx = width * 2 + origFreqReduced;
    const idxData = new Float32Array(numIdx);
    for (let i = 0; i < numIdx; i++) {
        idxData[i] = (i - width) / origFreqReduced;
    }
    // idx shape: [1, 1, numIdx]
    const idx = new Tensor(idxData, { shape: [1, 1, numIdx], dtype: 'float32' });

    // 生成相位偏移: [0, -1/newFreq, -2/newFreq, ..., -(newFreq-1)/newFreq]
    // t shape: [newFreqReduced, 1, 1]
    const phaseData = new Float32Array(newFreqReduced);
    for (let i = 0; i < newFreqReduced; i++) {
        phaseData[i] = -i / newFreqReduced;
    }
    const phase = new Tensor(phaseData, { shape: [newFreqReduced, 1, 1], dtype: 'float32' });

    // t = phase + idx  broadcast to [newFreqReduced, 1, numIdx]
    let t = k.add(phase, idx);

    // t *= baseFreq
    t = k.mul(t, baseFreq);

    // t = clamp(t, -lowpassFilterWidth, lowpassFilterWidth)
    t = k.clamp(t, -lowpassFilterWidth, lowpassFilterWidth);

    // 计算窗函数
    let window: Tensor;
    if (resamplingMethod === 'sinc_interp_hann') {
        // Hann 窗: cos²(πt / 2L)
        const piOverL = Math.PI / lowpassFilterWidth / 2;
        window = k.pow(k.cos(k.mul(t, piOverL)), 2);
    } else {
        // Kaiser 窗: I₀(β√(1-(t/L)²)) / I₀(β)
        const betaVal = beta ?? 14.769656459379492;
        const betaTensor = new Tensor([betaVal], { shape: [1], dtype: 'float32' });

        // 1 - (t / L)²
        const tNorm = k.div(t, lowpassFilterWidth);
        const tNormSq = k.mul(tNorm, tNorm);
        const one = new Tensor([1.0], { shape: [1], dtype: 'float32' });
        const oneMinusTNormSq = k.sub(one, tNormSq);

        // sqrt(max(0, 1 - (t/L)²))  防止数值误差导致负数
        const sqrtArg = k.clamp(oneMinusTNormSq, 0, undefined);
        const sqrtVal = k.sqrt(sqrtArg);

        // I₀(β * sqrt(...))
        const i0Arg = k.mul(betaTensor, sqrtVal);
        const i0Val = k.i0(i0Arg);

        // I₀(β)
        const i0Beta = k.i0(betaTensor);

        window = k.div(i0Val, i0Beta);
    }

    // 使用 sinc 算子 - normalized sinc: sin(πt)/(πt)
    // sinc 算子内部已处理 t=0 返回 1 的情况
    const sincVal = k.sinc(t);

    // 最终内核 = sinc * window * scale
    const scale = baseFreq / origFreqReduced;
    let kernel = k.mul(sincVal, window);
    kernel = k.mul(kernel, scale);

    // 重塑为 conv1d 权重格式: [out_channels, in_channels/groups, kernel_size]
    // 这里 out_channels = newFreqReduced, in_channels = 1
    // kernel shape 已经是 [newFreqReduced, 1, numIdx]

    return { kernel, width };
}

/**
 * 应用 sinc 重采样内核
 * 
 * 使用预计算的 kernel 执行重采样
 */
export function applySincResampleKernel(
    waveform: Tensor,
    origFreq: number,
    newFreq: number,
    gcdVal: number,
    kernel: Tensor,
    width: number,
): Tensor {
    const origFreqReduced = Math.floor(origFreq / gcdVal);
    const newFreqReduced = Math.floor(newFreq / gcdVal);

    // 保存原始形状
    const shape = waveform.shape;
    const length = shape[shape.length - 1];

    // 展平为 [batch, length]
    const batchDims = shape.slice(0, -1);
    const numWavs = batchDims.reduce((a, b) => a * b, 1);
    let flatWaveform = waveform.reshape([numWavs, length]);

    // Pad: (width, width + origFreqReduced)
    flatWaveform = k.pad(flatWaveform, [width, width + origFreqReduced], 'constant', 0);

    // 添加通道维度: [batch, 1, padded_length]
    const paddedLen = flatWaveform.shape[1];
    flatWaveform = flatWaveform.reshape([numWavs, 1, paddedLen]);

    // conv1d with stride = origFreqReduced
    // kernel shape: [newFreqReduced, 1, kernel_size]
    let resampled = k.conv1d(flatWaveform, kernel, undefined, origFreqReduced, 0, 1, 1);

    // resampled shape: [batch, newFreqReduced, num_frames]
    // 需要转置并 reshape 为 [batch, newFreqReduced * num_frames]
    // 然后截断到目标长度

    // transpose: [batch, num_frames, newFreqReduced]
    resampled = k.permute(resampled, [0, 2, 1]);

    // reshape: [batch, num_frames * newFreqReduced]
    // 注: reshape 内部会自动处理非连续张量 (ShapeHandler 会调用 contiguous)
    const numFrames = resampled.shape[1];
    resampled = resampled.reshape([numWavs, numFrames * newFreqReduced]);

    // 计算目标长度
    const targetLength = Math.ceil(newFreq * length / origFreq);

    // 截断到目标长度
    resampled = k.slice(resampled, `:, :${targetLength}`);

    // 恢复原始 batch 形状
    const outputShape = [...batchDims, targetLength];
    resampled = resampled.reshape(outputShape);

    return resampled;
}

/**
 * 使用带限 sinc 插值重采样波形
 * 
 * 对标 torchaudio.functional.resample
 * 
 * @param waveform - 输入波形张量 (..., time)
 * @param origFreq - 原始采样率 (Hz)
 * @param newFreq - 目标采样率 (Hz)
 * @param options - 重采样选项
 * @returns 重采样后的波形张量 (..., new_time)
 * 
 * @example
 * ```typescript
 * // 16kHz → 8kHz 下采样
 * const waveform = k.randn([1, 16000]);  // 1秒 16kHz
 * const resampled = resample(waveform, 16000, 8000);
 * console.log(resampled.shape);  // [1, 8000]
 * 
 * // 使用 Kaiser 窗
 * const resampled2 = resample(waveform, 16000, 22050, {
 *     resamplingMethod: 'sinc_interp_kaiser',
 *     beta: 14.769656459379492
 * });
 * ```
 */
export function resample(
    waveform: Tensor,
    origFreq: number,
    newFreq: number,
    options: ResampleOptions = {}
): Tensor {
    // 参数验证
    if (origFreq <= 0 || newFreq <= 0) {
        throw new Error('Original frequency and desired frequency should be positive');
    }

    // 频率必须是整数
    if (!Number.isInteger(origFreq) || !Number.isInteger(newFreq)) {
        throw new Error(
            'Frequencies must be of integer type to ensure quality resampling computation. ' +
            'To work around this, manually convert both frequencies to integer values ' +
            'that maintain their resampling rate ratio before passing them into the function.'
        );
    }

    // 相同频率直接返回
    if (origFreq === newFreq) {
        return waveform;
    }

    // 提取选项
    const {
        lowpass_filter_width = 6,
        rolloff = 0.99,
        resampling_method = 'sinc_interp_hann',
        beta,
    } = options;

    // 验证选项
    if (lowpass_filter_width <= 0) {
        throw new Error('Low pass filter width should be positive');
    }

    if (!['sinc_interp_hann', 'sinc_interp_kaiser'].includes(resampling_method)) {
        throw new Error(`Invalid resampling method: ${resampling_method}`);
    }

    // 计算 GCD
    const gcdVal = gcd(origFreq, newFreq);

    // 生成内核
    const { kernel, width } = getSincResampleKernel(
        origFreq,
        newFreq,
        gcdVal,
        lowpass_filter_width,
        rolloff,
        resampling_method,
        beta,
    );

    // 应用内核
    return applySincResampleKernel(waveform, origFreq, newFreq, gcdVal, kernel, width);
}
