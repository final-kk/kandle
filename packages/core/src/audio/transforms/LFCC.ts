/**
 * LFCC Transform
 *
 * 对标 torchaudio.transforms.LFCC
 *
 * Linear-Frequency Cepstral Coefficients
 */

import { type Tensor } from "../../tensor";
import { Module } from "../../nn/module";
import * as k from "../../index";
import { Spectrogram } from "./Spectrogram";
import { amplitudeToDB, linearFbanks, createDct } from "../functional";

export interface LFCCOptions {
    /** 采样率 (Hz) */
    sample_rate?: number;
    /** LFCC 系数数量 */
    n_lfcc?: number;
    /** DCT 类型 (目前只支持 2) */
    dct_type?: number;
    /** DCT 归一化方式 */
    dct_norm?: "ortho" | null;
    /** 是否使用 log 而非 dB */
    log_lf?: boolean;
    /** 频谱图参数 */
    speckwargs?: {
        n_fft?: number;
        win_length?: number | null;
        hop_length?: number | null;
        pad?: number;
        window_fn?: "hann" | "hamming" | "blackman";
        center?: boolean;
        pad_mode?: "constant" | "reflect" | "replicate" | "circular";
        onesided?: boolean;
    };
    /** 线性滤波器数量 */
    n_filter?: number;
    /** 最小频率 (Hz) */
    f_min?: number;
    /** 最大频率 (Hz) */
    f_max?: number | null;
}

/**
 * LFCC 变换类
 *
 * 计算线性频率倒谱系数
 *
 * @example
 * ```ts
 * const lfcc = new LFCC({
 *     sample_rate: 16000,
 *     n_lfcc: 13,
 *     n_filter: 128
 * });
 * const coeffs = await lfcc.forward(waveform);
 * ```
 */
export class LFCC extends Module {
    private spectrogram!: Spectrogram;
    private n_lfcc: number;
    private n_filter: number;
    private sample_rate: number;
    private f_min: number;
    private f_max: number;
    private n_stft: number;
    private dct_norm: "ortho" | null;
    private log_lf: boolean;
    private fb!: Tensor;
    private dctMat!: Tensor;

    constructor(options: LFCCOptions = {}) {
        super();

        this.sample_rate = options.sample_rate ?? 16000;
        this.n_lfcc = options.n_lfcc ?? 40;
        this.n_filter = options.n_filter ?? 128;
        this.f_min = options.f_min ?? 0.0;
        this.f_max = options.f_max ?? this.sample_rate / 2;
        this.dct_norm = options.dct_norm ?? "ortho";
        this.log_lf = options.log_lf ?? false;

        if (options.dct_type !== undefined && options.dct_type !== 2) {
            throw new Error("LFCC: Only DCT type 2 is supported");
        }

        const speckwargs = options.speckwargs ?? {};
        const n_fft = speckwargs.n_fft ?? 400;
        const onesided = speckwargs.onesided ?? true;
        this.n_stft = onesided ? Math.floor(n_fft / 2) + 1 : n_fft;

        // 创建 Spectrogram
        this.spectrogram = new Spectrogram({
            n_fft,
            win_length: speckwargs.win_length ?? n_fft,
            hop_length: speckwargs.hop_length ?? Math.floor(n_fft / 4),
            pad: speckwargs.pad ?? 0,
            window_fn: speckwargs.window_fn ?? "hann",
            power: 2.0, // 使用功率谱
            normalized: false,
            center: speckwargs.center ?? true,
            pad_mode: speckwargs.pad_mode ?? "reflect",
            onesided,
        });

        // 创建线性滤波器组
        this.fb = linearFbanks(
            this.n_stft,
            this.f_min,
            this.f_max,
            this.n_filter,
            this.sample_rate
        );

        // 创建 DCT 矩阵
        this.dctMat = createDct(this.n_lfcc, this.n_filter, this.dct_norm);

        // 注册子模块
        this.addModule("spectrogram", this.spectrogram);

        // 注册 buffers
        this.registerBuffer("fb", this.fb);
        this.registerBuffer("dctMat", this.dctMat);
    }

    /**
     * 计算 LFCC
     *
     * @param waveform - 输入波形 (..., time)
     * @returns LFCC 系数 (..., n_lfcc, time)
     */
    async forward(waveform: Tensor): Promise<Tensor> {
        // Step 1: Spectrogram
        const spec = await this.spectrogram.forward(waveform);

        // Step 2: Linear filterbank
        const shape = spec.shape;
        const ndim = shape.length;

        // spec: (..., n_stft, T), fb: (n_stft, n_filter)
        // -> (..., n_filter, T)
        const transposed = k.transpose(spec, ndim - 2, ndim - 1);
        const linearSpec = k.matmul(transposed, this.fb);
        const linearSpecT = k.transpose(linearSpec, ndim - 2, ndim - 1);

        // Step 3: Log/dB scale
        let logLinearSpec: Tensor;
        if (this.log_lf) {
            logLinearSpec = k.log(k.clamp(linearSpecT, 1e-10));
        } else {
            logLinearSpec = amplitudeToDB(linearSpecT, 10.0, 1e-10, 1.0);
        }

        // Step 4: DCT
        // logLinearSpec: (..., n_filter, T)
        // dctMat: (n_filter, n_lfcc)
        // -> (..., n_lfcc, T)
        const transposed2 = k.transpose(logLinearSpec, ndim - 2, ndim - 1);
        const result = k.matmul(transposed2, this.dctMat);
        return k.transpose(result, ndim - 2, ndim - 1);
    }
}
