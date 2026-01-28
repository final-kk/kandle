/**
 * MFCC Transform
 *
 * 对标 torchaudio.transforms.MFCC
 *
 * Mel-Frequency Cepstral Coefficients
 */

import { type Tensor } from "../../tensor";
import { Module } from "../../nn/module";
import * as k from "../../index";
import { MelSpectrogram, type MelSpectrogramOptions } from "./MelSpectrogram";
import { amplitudeToDB, createDct } from "../functional";

export interface MFCCOptions extends MelSpectrogramOptions {
    /** MFCC 系数数量 */
    n_mfcc?: number;
    /** DCT 类型 (目前只支持 2) */
    dct_type?: number;
    /** DCT 归一化方式 */
    dct_norm?: "ortho" | null;
    /** 是否使用 log-mel (而非 dB) */
    log_mels?: boolean;
}

/**
 * MFCC 变换类
 *
 * 计算 Mel 频率倒谱系数
 *
 * @example
 * ```ts
 * const mfcc = new MFCC({
 *     sample_rate: 16000,
 *     n_mfcc: 13,
 *     n_mels: 40
 * });
 * const coeffs = await mfcc.forward(waveform);
 * ```
 */
export class MFCC extends Module {
    private melSpectrogram!: MelSpectrogram;
    private n_mfcc: number;
    private n_mels: number;
    private dct_norm: "ortho" | null;
    private log_mels: boolean;
    private dctMat!: Tensor;

    constructor(options: MFCCOptions = {}) {
        super();

        this.n_mfcc = options.n_mfcc ?? 40;
        this.n_mels = options.n_mels ?? 128;
        this.dct_norm = options.dct_norm ?? "ortho";
        this.log_mels = options.log_mels ?? false;

        if (options.dct_type !== undefined && options.dct_type !== 2) {
            throw new Error("MFCC: Only DCT type 2 is supported");
        }

        // 创建 MelSpectrogram
        this.melSpectrogram = new MelSpectrogram({
            ...options,
            n_mels: this.n_mels,
            power: 2.0, // 使用功率谱
        });

        // 创建 DCT 矩阵
        this.dctMat = createDct(this.n_mfcc, this.n_mels, this.dct_norm);

        // 注册子模块
        this.addModule("melSpectrogram", this.melSpectrogram);

        // 注册 buffer
        this.registerBuffer("dctMat", this.dctMat);
    }

    /**
     * 计算 MFCC
     *
     * @param waveform - 输入波形 (..., time)
     * @returns MFCC 系数 (..., n_mfcc, time)
     */
    async forward(waveform: Tensor): Promise<Tensor> {
        // Step 1: Mel Spectrogram
        const melSpec = await this.melSpectrogram.forward(waveform);

        // Step 2: Log/dB scale
        let logMelSpec: Tensor;
        if (this.log_mels) {
            // 使用自然对数
            logMelSpec = k.log(k.clamp(melSpec, 1e-10));
        } else {
            // 使用 dB 刻度 (默认)
            logMelSpec = amplitudeToDB(melSpec, 10.0, 1e-10, 1.0);
        }

        // Step 3: DCT
        // logMelSpec: (..., n_mels, T)
        // dctMat: (n_mels, n_mfcc)
        // 输出: (..., n_mfcc, T)

        const shape = logMelSpec.shape;
        const ndim = shape.length;

        // Transpose: (..., n_mels, T) -> (..., T, n_mels)
        const transposed = k.transpose(logMelSpec, ndim - 2, ndim - 1);

        // matmul: (..., T, n_mels) @ (n_mels, n_mfcc) -> (..., T, n_mfcc)
        const result = k.matmul(transposed, this.dctMat);

        // Transpose back: (..., T, n_mfcc) -> (..., n_mfcc, T)
        return k.transpose(result, ndim - 2, ndim - 1);
    }
}
