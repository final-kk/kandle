/**
 * TimeStretch Transform
 *
 * 对标 torchaudio.transforms.TimeStretch
 *
 * 在时间维度拉伸 STFT 而不改变音高
 */

import { type Tensor } from "../../tensor";
import { Module } from "../../nn/module";
import * as k from "../../index";
import { phaseVocoder } from "../functional";

export interface TimeStretchOptions {
    /** 帧移 (默认: 从 n_freq 推导) */
    hop_length?: number | null;
    /** 频率 bin 数，即 n_fft // 2 + 1 (默认: 201) */
    n_freq?: number;
    /** 固定拉伸率 (可在 forward 时覆盖) */
    fixed_rate?: number | null;
}

/**
 * 时间拉伸转换类
 *
 * 使用 Phase Vocoder 算法在时间维度拉伸复数 STFT 频谱图，
 * 不改变音高。
 *
 * @example
 * ```ts
 * const transform = new TimeStretch({ n_freq: 257 });
 * // 减速 50%
 * const stretched = await transform.forward(stftResult, 0.5);
 * ```
 */
export class TimeStretch extends Module {
    private hop_length: number;
    private n_freq: number;
    private fixed_rate: number | null;
    private phaseAdvance!: Tensor;

    constructor(options: TimeStretchOptions = {}) {
        super();

        this.n_freq = options.n_freq ?? 201;
        // hop_length 默认: (n_freq - 1) * 2 / 4 = n_fft / 4
        // 对应 n_fft = (n_freq - 1) * 2
        const n_fft = (this.n_freq - 1) * 2;
        this.hop_length = options.hop_length ?? Math.floor(n_fft / 4);
        this.fixed_rate = options.fixed_rate ?? null;

        // 预计算 phaseAdvance: linspace(0, pi*hop_length, n_freq)
        this.phaseAdvance = k.linspace(0, Math.PI * this.hop_length, this.n_freq, "float32");

        // 注册 buffer
        this.registerBuffer("phaseAdvance", this.phaseAdvance);
    }

    /**
     * 时间拉伸复数频谱图
     *
     * @param complexSpecgrams - 复数 STFT 频谱图 (..., freq, time)
     * @param rate - 拉伸率 (覆盖 fixed_rate)。rate > 1 加速，rate < 1 减速
     * @returns 拉伸后的复数频谱图 (..., freq, ceil(time / rate))
     */
    async forward(complex_specgrams: Tensor, rate?: number): Promise<Tensor> {
        const actualRate = rate ?? this.fixed_rate;
        if (actualRate === null) {
            throw new Error(
                "TimeStretch: rate must be provided either in constructor (fixed_rate) or forward()"
            );
        }
        return phaseVocoder(complex_specgrams, actualRate, this.phaseAdvance);
    }
}
