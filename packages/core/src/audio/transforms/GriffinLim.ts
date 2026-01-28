/**
 * GriffinLim Transform
 *
 * 对标 torchaudio.transforms.GriffinLim
 *
 * 使用 Griffin-Lim 算法从幅度频谱图重建波形
 */

import { type Tensor } from "../../tensor";
import { Module } from "../../nn/module";
import * as k from "../../index";
import { griffinlim as griffinlimFn } from "../functional";

export interface GriffinLimOptions {
    /** FFT 大小 (默认: 400) */
    n_fft?: number;
    /** 迭代次数 (默认: 32) */
    n_iter?: number;
    /** 窗长度 (默认: n_fft) */
    win_length?: number | null;
    /** 帧移 (默认: win_length // 2) */
    hop_length?: number | null;
    /** 窗函数类型 */
    window_fn?: "hann" | "hamming" | "blackman";
    /** 输入谱图功率指数 (默认: 2.0) */
    power?: number;
    /** 快速 Griffin-Lim 动量 (默认: 0.99) */
    momentum?: number;
    /** 输出波形长度 */
    length?: number | null;
    /** 是否随机初始化相位 (默认: true) */
    rand_init?: boolean;
}

/**
 * Griffin-Lim 转换类
 *
 * 从线性幅度/功率频谱图重建波形。
 *
 * @example
 * ```ts
 * const transform = new GriffinLim({ n_fft: 400, n_iter: 32 });
 * const waveform = await transform.forward(spectrogram);
 * ```
 */
export class GriffinLim extends Module {
    private n_fft: number;
    private n_iter: number;
    private win_length: number;
    private hop_length: number;
    private power: number;
    private momentum: number;
    private length: number | null;
    private rand_init: boolean;
    private window!: Tensor;

    constructor(options: GriffinLimOptions = {}) {
        super();

        this.n_fft = options.n_fft ?? 400;
        this.n_iter = options.n_iter ?? 32;
        this.win_length = options.win_length ?? this.n_fft;
        this.hop_length = options.hop_length ?? Math.floor(this.win_length / 2);
        this.power = options.power ?? 2.0;
        this.momentum = options.momentum ?? 0.99;
        this.length = options.length ?? null;
        this.rand_init = options.rand_init ?? true;

        // 创建窗函数
        const windowFn = options.window_fn ?? "hann";
        switch (windowFn) {
            case "hann":
                this.window = k.hannWindow(this.win_length);
                break;
            case "hamming":
                this.window = k.hammingWindow(this.win_length);
                break;
            case "blackman":
                this.window = k.blackmanWindow(this.win_length);
                break;
            default:
                this.window = k.hannWindow(this.win_length);
        }

        // 注册 buffer
        this.registerBuffer("window", this.window);
    }

    /**
     * 从幅度/功率频谱图重建波形
     *
     * @param specgram - 幅度/功率频谱图 (..., freq, time)
     * @param length - 输出波形长度 (覆盖构造时的 length)
     * @returns 重建的波形 (..., time)
     */
    async forward(specgram: Tensor, length?: number): Promise<Tensor> {
        const outputLength = length ?? this.length ?? undefined;
        return griffinlimFn(
            specgram,
            this.window,
            this.n_fft,
            this.hop_length,
            this.win_length,
            this.power,
            this.n_iter,
            this.momentum,
            outputLength,
            this.rand_init
        );
    }
}
