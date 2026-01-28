/**
 * PitchShift Transform
 *
 * 对标 torchaudio.transforms.PitchShift
 *
 * 改变音频的音高而不改变时长
 * Transform 类会预计算所有可缓存参数，适合多次变换相同参数的场景
 */

import { type Tensor } from "../../tensor";
import { Module } from "../../nn/module";
import * as k from "../../index";
import { phaseVocoder } from "../functional";
import { Resample } from "./Resample";

export interface PitchShiftTransformOptions {
    /** 采样率 (Hz) @default 16000 */
    sample_rate?: number;
    /** 音高变换步数 (正数升高, 负数降低) */
    n_steps: number;
    /** 每八度的音阶数 (默认: 12, 半音) */
    bins_per_octave?: number;
    /** FFT 大小 (默认: 512) */
    n_fft?: number;
    /** 窗口长度 (默认: n_fft) */
    win_length?: number;
    /** 帧移长度 (默认: win_length // 4) */
    hop_length?: number;
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
 */
function approximateRate(
    rate: number,
    maxDenominator: number = 64
): { numerator: number; denominator: number } {
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

        if (error < 0.001) break;
    }

    const g = gcd(bestNum, bestDen);
    return { numerator: bestNum / g, denominator: bestDen / g };
}

/**
 * 音高变换转换类
 *
 * 使用 Phase Vocoder + Resample 在不改变时长的情况下改变音高。
 * 与 functional.pitchShift 不同，Transform 类在构造时预计算所有参数，
 * forward 调用时直接执行流水线，避免重复计算开销。
 *
 * @example
 * ```ts
 * // 升高一个八度 (12 个半音)
 * const transform = new PitchShift({ sample_rate: 16000, n_steps: 12 });
 * const higher = await transform.forward(waveform);
 *
 * // 降低一个全音 (2 个半音)
 * const transform2 = new PitchShift({ sample_rate: 16000, n_steps: -2 });
 * const lower = await transform2.forward(waveform);
 * ```
 */
export class PitchShift extends Module {
    private sample_rate: number;
    private n_steps: number;
    private n_fft: number;
    private win_length: number;
    private hop_length: number;

    // 缓存的参数
    private approxRate: number;
    private window!: Tensor;
    private phaseAdvance!: Tensor;
    private resampler!: Resample | null;

    constructor(options: PitchShiftTransformOptions) {
        super();

        this.sample_rate = options.sample_rate ?? 16000;
        this.n_steps = options.n_steps;

        const bins_per_octave = options.bins_per_octave ?? 12;
        this.n_fft = options.n_fft ?? 512;
        this.win_length = options.win_length ?? this.n_fft;
        this.hop_length = options.hop_length ?? Math.floor(this.win_length / 4);

        // 如果 n_steps 为 0，不需要任何预计算
        if (this.n_steps === 0) {
            this.approxRate = 1;
            this.window = null as unknown as Tensor;
            this.phaseAdvance = null as unknown as Tensor;
            this.resampler = null;
        } else {
            // 预计算 rate
            const rate = Math.pow(2, this.n_steps / bins_per_octave);
            const { numerator, denominator } = approximateRate(rate, 64);
            this.approxRate = numerator / denominator;

            // 预计算 window
            this.window = k.hannWindow(this.win_length);

            // 预计算 phaseAdvance
            const numFreqs = this.n_fft / 2 + 1;
            this.phaseAdvance = k.linspace(0, Math.PI * this.hop_length, numFreqs, "float32");

            // 预实例化 Resample Transform
            const origFreq = this.sample_rate * denominator;
            const newFreq = this.sample_rate * numerator;
            this.resampler = new Resample({ orig_freq: origFreq, new_freq: newFreq });

            // 注册 buffers
            this.registerBuffer("window", this.window);
            this.registerBuffer("phaseAdvance", this.phaseAdvance);

            // 注册子模块
            this.addModule("resampler", this.resampler);
        }
    }

    /**
     * 变换音高
     *
     * @param waveform - 输入波形 (..., time)
     * @returns 变换后的波形 (..., time)
     */
    async forward(waveform: Tensor): Promise<Tensor> {
        // n_steps 为 0，直接返回
        if (this.n_steps === 0) {
            return waveform;
        }

        // 保存原始形状
        const origShape = waveform.shape;
        const origLength = origShape[origShape.length - 1];

        // 展平为 (batch, time)
        let flat = waveform;
        if (origShape.length === 1) {
            flat = k.unsqueeze(waveform, 0);
        } else if (origShape.length > 2) {
            const batchSize = origShape.slice(0, -1).reduce((a, b) => a * b, 1);
            flat = k.reshape(waveform, [batchSize, origLength]);
        }

        // Step 1: STFT
        const stftResult = k.stft(
            flat,
            this.n_fft,
            this.hop_length,
            this.win_length,
            this.window, // 使用缓存的 window
            true, // center
            "reflect", // pad_mode
            false, // normalized
            true, // onesided
            true // return_complex
        );

        // Step 2: Phase Vocoder (时间拉伸) - 使用缓存的 phaseAdvance
        const stretched = phaseVocoder(stftResult, this.approxRate, this.phaseAdvance);

        // Step 3: iSTFT (重建波形) - 使用缓存的 window
        const stretchedWaveform = k.istft(
            stretched,
            this.n_fft,
            this.hop_length,
            this.win_length,
            this.window, // 使用缓存的 window
            true, // center
            false, // normalized
            true, // onesided
            undefined // length
        );

        // Step 4: Resample - 使用缓存的 Resample Transform
        const resampled = await this.resampler!.forward(stretchedWaveform);

        // Step 5: 裁剪或填充到原始长度
        const resampledLength = resampled.shape[resampled.shape.length - 1];
        let result: Tensor;

        if (resampledLength === origLength) {
            result = resampled;
        } else if (resampledLength > origLength) {
            result = k.slice(resampled, `..., :${origLength}`);
        } else {
            const padAmount = origLength - resampledLength;
            result = k.pad(resampled, [0, padAmount], "constant", 0);
        }

        // 恢复原始形状
        if (origShape.length === 1) {
            result = k.squeeze(result, 0);
        } else if (origShape.length > 2) {
            result = k.reshape(result, [...origShape] as number[]);
        }

        return result;
    }
}
