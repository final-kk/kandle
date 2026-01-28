/**
 * Spectrogram Transform
 *
 * 对标 torchaudio.transforms.Spectrogram 和 InverseSpectrogram
 */

import { type Tensor } from "../../tensor";
import { Module } from "../../nn/module";
import * as k from "../../index";
import {
    spectrogram as spectrogramFn,
    inverseSpectrogram as inverseSpectrogramFn,
} from "../functional";

export interface SpectrogramOptions {
    /** FFT 大小 */
    n_fft?: number;
    /** 窗长度 (默认: n_fft) */
    win_length?: number | null;
    /** 帧移 (默认: win_length // 2) */
    hop_length?: number | null;
    /** 两侧 padding 采样点数 */
    pad?: number;
    /** 窗函数类型 */
    window_fn?: "hann" | "hamming" | "blackman";
    /** 功率指数: null=复数, 1=幅度, 2=功率 */
    power?: number | null;
    /** 是否归一化 */
    normalized?: boolean;
    /** 是否中心 padding */
    center?: boolean;
    /** padding 模式 */
    pad_mode?: "constant" | "reflect" | "replicate" | "circular";
    /** 是否只返回正频率 */
    onesided?: boolean;
}

/**
 * 频谱图变换类
 *
 * @example
 * ```ts
 * const transform = new Spectrogram({ n_fft: 400, power: 2 });
 * const spec = await transform.forward(waveform);
 * ```
 */
export class Spectrogram extends Module {
    private n_fft: number;
    private win_length: number;
    private hop_length: number;
    private pad: number;
    private power: number | null;
    private normalized: boolean;
    private center: boolean;
    private pad_mode: "constant" | "reflect" | "replicate" | "circular";
    private onesided: boolean;
    private window!: Tensor;

    constructor(options: SpectrogramOptions = {}) {
        super();

        this.n_fft = options.n_fft ?? 400;
        this.win_length = options.win_length ?? this.n_fft;
        this.hop_length = options.hop_length ?? Math.floor(this.win_length / 2);
        this.pad = options.pad ?? 0;
        this.power = options.power ?? null;
        this.normalized = options.normalized ?? false;
        this.center = options.center ?? true;
        this.pad_mode = options.pad_mode ?? "reflect";
        this.onesided = options.onesided ?? true;

        // 创建窗函数
        const windowFn = options.window_fn ?? "hann";
        let window: Tensor;
        switch (windowFn) {
            case "hann":
                window = k.hannWindow(this.win_length);
                break;
            case "hamming":
                window = k.hammingWindow(this.win_length);
                break;
            case "blackman":
                window = k.blackmanWindow(this.win_length);
                break;
            default:
                window = k.hannWindow(this.win_length);
        }
        this.window = window;
        this.registerBuffer("window", window);
    }

    /**
     * 将波形转换为频谱图
     *
     * @param waveform - 输入波形 (..., time)
     * @returns 频谱图 (..., n_freqs, time)
     */
    async forward(waveform: Tensor): Promise<Tensor> {
        return spectrogramFn(
            waveform,
            this.pad,
            this.window,
            this.n_fft,
            this.hop_length,
            this.win_length,
            this.power,
            this.normalized,
            this.center,
            this.pad_mode,
            this.onesided
        );
    }
}

export interface InverseSpectrogramOptions {
    /** FFT 大小 */
    n_fft?: number;
    /** 窗长度 */
    win_length?: number | null;
    /** 帧移 */
    hop_length?: number | null;
    /** 两侧 padding 采样点数 */
    pad?: number;
    /** 窗函数类型 */
    window_fn?: "hann" | "hamming" | "blackman";
    /** 是否归一化 */
    normalized?: boolean;
    /** 是否使用 center padding */
    center?: boolean;
    /** 是否为单边频谱 */
    onesided?: boolean;
}

/**
 * 逆频谱图变换类
 *
 * @example
 * ```ts
 * const transform = new InverseSpectrogram({ n_fft: 400 });
 * const waveform = await transform.forward(spectrogram);
 * ```
 */
export class InverseSpectrogram extends Module {
    private n_fft: number;
    private win_length: number;
    private hop_length: number;
    private pad: number;
    private normalized: boolean;
    private center: boolean;
    private onesided: boolean;
    private window!: Tensor;

    constructor(options: InverseSpectrogramOptions = {}) {
        super();

        this.n_fft = options.n_fft ?? 400;
        this.win_length = options.win_length ?? this.n_fft;
        this.hop_length = options.hop_length ?? Math.floor(this.win_length / 2);
        this.pad = options.pad ?? 0;
        this.normalized = options.normalized ?? false;
        this.center = options.center ?? true;
        this.onesided = options.onesided ?? true;

        // 创建窗函数
        const windowFn = options.window_fn ?? "hann";
        let window: Tensor;
        switch (windowFn) {
            case "hann":
                window = k.hannWindow(this.win_length);
                break;
            case "hamming":
                window = k.hammingWindow(this.win_length);
                break;
            case "blackman":
                window = k.blackmanWindow(this.win_length);
                break;
            default:
                window = k.hannWindow(this.win_length);
        }
        this.window = window;
        this.registerBuffer("window", window);
    }

    /**
     * 从频谱图重建波形
     *
     * @param spec - 复数频谱图 (..., n_freqs, time)
     * @param length - 输出波形长度 (可选)
     * @returns 重建的波形 (..., time)
     */
    async forward(spec: Tensor, length?: number): Promise<Tensor> {
        return inverseSpectrogramFn(
            spec,
            length,
            this.pad,
            this.window,
            this.n_fft,
            this.hop_length,
            this.win_length,
            this.normalized,
            this.center,
            this.onesided
        );
    }
}
