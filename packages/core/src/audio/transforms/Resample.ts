/**
 * Resample Transform
 *
 * 对标 torchaudio.transforms.Resample
 *
 * 使用带限 sinc 插值重采样波形
 * Transform 类会预计算并缓存重采样内核，适合多次重采样相同参数的场景
 */

import { type Tensor } from "../../tensor";
import { Module } from "../../nn/module";
import {
    gcd,
    getSincResampleKernel,
    applySincResampleKernel,
    type ResampleOptions,
} from "../functional";

export interface ResampleTransformOptions extends ResampleOptions {
    /** 原始采样率 (Hz) @default 16000 */
    orig_freq?: number;
    /** 目标采样率 (Hz) @default 16000 */
    new_freq?: number;
}

/**
 * 重采样转换类
 *
 * 使用带限 sinc 插值改变波形采样率。
 * 与 functional.resample 不同，Transform 类在构造时预计算内核，
 * forward 调用时直接应用，避免重复计算开销。
 *
 * @example
 * ```ts
 * // 16kHz → 8kHz 下采样
 * const transform = new Resample({ orig_freq: 16000, new_freq: 8000 });
 * const resampled = await transform.forward(waveform);
 *
 * // 使用 Kaiser 窗
 * const transform2 = new Resample({
 *     orig_freq: 44100,
 *     new_freq: 22050,
 *     resampling_method: 'sinc_interp_kaiser',
 *     beta: 14.769656459379492
 * });
 * ```
 */
export class Resample extends Module {
    private orig_freq: number;
    private new_freq: number;
    private gcdVal: number;
    private kernel!: Tensor;
    private width: number;

    constructor(options: ResampleTransformOptions = {}) {
        super();

        this.orig_freq = options.orig_freq ?? 16000;
        this.new_freq = options.new_freq ?? 16000;

        // 参数验证
        if (this.orig_freq <= 0 || this.new_freq <= 0) {
            throw new Error("Original frequency and desired frequency should be positive");
        }
        if (!Number.isInteger(this.orig_freq) || !Number.isInteger(this.new_freq)) {
            throw new Error("Frequencies must be of integer type");
        }

        // 计算 GCD
        this.gcdVal = gcd(this.orig_freq, this.new_freq);

        // 相同频率时使用 identity kernel
        if (this.orig_freq === this.new_freq) {
            // 创建一个简单的 identity kernel，不会被实际使用
            // forward 会直接返回输入
            this.kernel = null as unknown as Tensor;
            this.width = 0;
        } else {
            // 预计算 kernel
            const {
                lowpass_filter_width = 6,
                rolloff = 0.99,
                resampling_method = "sinc_interp_hann",
                beta,
            } = options;

            if (lowpass_filter_width <= 0) {
                throw new Error("Low pass filter width should be positive");
            }

            const result = getSincResampleKernel(
                this.orig_freq,
                this.new_freq,
                this.gcdVal,
                lowpass_filter_width,
                rolloff,
                resampling_method,
                beta
            );
            this.kernel = result.kernel;
            this.width = result.width;

            // 注册 buffer
            this.registerBuffer("kernel", this.kernel);
        }
    }

    /**
     * 重采样波形
     *
     * @param waveform - 输入波形 (..., time)
     * @returns 重采样后的波形 (..., new_time)
     */
    async forward(waveform: Tensor): Promise<Tensor> {
        // 相同频率直接返回
        if (this.orig_freq === this.new_freq) {
            return waveform;
        }

        // 使用预计算的 kernel 执行重采样
        return applySincResampleKernel(
            waveform,
            this.orig_freq,
            this.new_freq,
            this.gcdVal,
            this.kernel,
            this.width
        );
    }
}
