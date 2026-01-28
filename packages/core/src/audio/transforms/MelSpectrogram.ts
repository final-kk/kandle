/**
 * MelSpectrogram Transform
 *
 * 对标 torchaudio.transforms.MelSpectrogram
 *
 * 组合 Spectrogram + MelScale
 */

import { type Tensor } from "../../tensor";
import { Module } from "../../nn/module";
import { Spectrogram } from "./Spectrogram";
import { MelScale } from "./MelScale";
import { type MelScaleType, type NormType } from "../functional";

export interface MelSpectrogramOptions {
    /** 采样率 (Hz) */
    sample_rate?: number;
    /** FFT 大小 */
    n_fft?: number;
    /** 窗长度 */
    win_length?: number | null;
    /** 帧移 */
    hop_length?: number | null;
    /** 最小频率 (Hz) */
    f_min?: number;
    /** 最大频率 (Hz) */
    f_max?: number | null;
    /** 两侧 padding */
    pad?: number;
    /** Mel 滤波器数量 */
    n_mels?: number;
    /** 窗函数类型 */
    window_fn?: "hann" | "hamming" | "blackman";
    /** 功率指数 */
    power?: number;
    /** 是否归一化 STFT */
    normalized?: boolean;
    /** 窗函数参数 (保留) */
    wkwargs?: Record<string, any> | null;
    /** 是否中心 padding */
    center?: boolean;
    /** padding 模式 */
    pad_mode?: "constant" | "reflect" | "replicate" | "circular";
    /** 是否单边频谱 */
    onesided?: boolean | null;
    /** Mel 归一化方式 */
    norm?: NormType;
    /** Mel 频率转换公式 */
    mel_scale?: MelScaleType;
}

/**
 * Mel 频谱图变换类
 *
 * 将音频波形直接转换为 Mel 频谱图
 *
 * @example
 * ```ts
 * const transform = new MelSpectrogram({
 *     sample_rate: 16000,
 *     n_fft: 400,
 *     n_mels: 80
 * });
 * const melSpec = await transform.forward(waveform);
 * ```
 */
export class MelSpectrogram extends Module {
    private spectrogram!: Spectrogram;
    private melScale!: MelScale;

    constructor(options: MelSpectrogramOptions = {}) {
        super();

        const sample_rate = options.sample_rate ?? 16000;
        const n_fft = options.n_fft ?? 400;
        const win_length = options.win_length ?? n_fft;
        const hop_length = options.hop_length ?? Math.floor(win_length / 2);
        const n_mels = options.n_mels ?? 128;
        const f_min = options.f_min ?? 0.0;
        const f_max = options.f_max ?? sample_rate / 2;
        const power = options.power ?? 2.0;
        const onesided = options.onesided ?? true;

        // 创建 Spectrogram
        this.spectrogram = new Spectrogram({
            n_fft,
            win_length,
            hop_length,
            pad: options.pad ?? 0,
            window_fn: options.window_fn ?? "hann",
            power,
            normalized: options.normalized ?? false,
            center: options.center ?? true,
            pad_mode: options.pad_mode ?? "reflect",
            onesided,
        });

        // 计算 n_stft
        const n_stft = onesided ? Math.floor(n_fft / 2) + 1 : n_fft;

        // 创建 MelScale
        this.melScale = new MelScale({
            n_mels,
            sample_rate,
            f_min,
            f_max,
            n_stft,
            norm: options.norm ?? null,
            mel_scale: options.mel_scale ?? "htk",
        });

        // 注册子模块
        this.addModule("spectrogram", this.spectrogram);
        this.addModule("melScale", this.melScale);
    }

    /**
     * 将波形转换为 Mel 频谱图
     *
     * @param waveform - 输入波形 (..., time)
     * @returns Mel 频谱图 (..., n_mels, time)
     */
    async forward(waveform: Tensor): Promise<Tensor> {
        // Step 1: Spectrogram
        const spec = await this.spectrogram.forward(waveform);

        // Step 2: MelScale
        return await this.melScale.forward(spec);
    }
}
