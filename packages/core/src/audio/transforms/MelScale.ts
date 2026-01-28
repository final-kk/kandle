/**
 * MelScale Transform
 *
 * 对标 torchaudio.transforms.MelScale 和 InverseMelScale
 */

import { type Tensor } from "../../tensor";
import { Module } from "../../nn/module";
import * as k from "../../index";
import { melscaleFbanks, type MelScaleType, type NormType } from "../functional";

export interface MelScaleOptions {
    /** Mel 滤波器数量 */
    n_mels?: number;
    /** 采样率 (Hz) */
    sample_rate?: number;
    /** 最小频率 (Hz) */
    f_min?: number;
    /** 最大频率 (Hz) */
    f_max?: number | null;
    /** STFT 输出频率 bins 数量 */
    n_stft?: number;
    /** 归一化方式 */
    norm?: NormType;
    /** Mel 频率转换公式 */
    mel_scale?: MelScaleType;
}

/**
 * Mel 频率尺度变换
 *
 * 将普通频谱图转换为 Mel 频率频谱图
 *
 * @example
 * ```ts
 * const melScale = new MelScale({ n_mels: 80, sample_rate: 16000 });
 * const melSpec = await melScale.forward(spectrogram);
 * ```
 */
export class MelScale extends Module {
    private n_mels: number;
    private sample_rate: number;
    private f_min: number;
    private f_max: number;
    private n_stft: number;
    private norm: NormType;
    private mel_scale: MelScaleType;
    private fb!: Tensor;

    constructor(options: MelScaleOptions = {}) {
        super();

        this.n_mels = options.n_mels ?? 128;
        this.sample_rate = options.sample_rate ?? 16000;
        this.f_min = options.f_min ?? 0.0;
        this.f_max = options.f_max ?? this.sample_rate / 2;
        this.n_stft = options.n_stft ?? 201; // 默认假设 n_fft=400
        this.norm = options.norm ?? null;
        this.mel_scale = options.mel_scale ?? "htk";

        // 直接创建滤波器组矩阵
        const fb = melscaleFbanks(
            this.n_stft,
            this.f_min,
            this.f_max,
            this.n_mels,
            this.sample_rate,
            this.norm,
            this.mel_scale
        );
        this.fb = fb;
        this.registerBuffer("fb", fb);
    }

    /**
     * 将频谱图转换为 Mel 频谱图
     *
     * @param specgram - 输入频谱图 (..., n_stft, time)
     * @returns Mel 频谱图 (..., n_mels, time)
     */
    async forward(specgram: Tensor): Promise<Tensor> {
        // specgram: (..., n_stft, time)
        // fb: (n_stft, n_mels)
        // 输出: (..., n_mels, time)

        // 需要: specgram^T @ fb 沿最后两个维度
        // 或者: (fb^T @ specgram) 然后 transpose

        // 使用 einsum 或 matmul:
        // specgram.shape = (..., F, T), fb.shape = (F, M)
        // 我们想要: (..., M, T) = einsum('...ft,fm->...mt', specgram, fb)

        // 由于没有直接的 einsum，使用 permute + matmul
        // specgram: (..., F, T) -> (..., T, F)
        // matmul: (..., T, F) @ (F, M) -> (..., T, M)
        // -> (..., M, T)

        const shape = specgram.shape;
        const ndim = shape.length;

        // Transpose last two dims: (..., F, T) -> (..., T, F)
        const transposed = k.transpose(specgram, ndim - 2, ndim - 1);

        // matmul: (..., T, F) @ (F, M) -> (..., T, M)
        const result = k.matmul(transposed, this.fb);

        // Transpose back: (..., T, M) -> (..., M, T)
        return k.transpose(result, ndim - 2, ndim - 1);
    }
}

export interface InverseMelScaleOptions {
    /** STFT 输出频率 bins 数量 */
    n_stft?: number;
    /** Mel 滤波器数量 */
    n_mels?: number;
    /** 采样率 (Hz) */
    sample_rate?: number;
    /** 最小频率 (Hz) */
    f_min?: number;
    /** 最大频率 (Hz) */
    f_max?: number | null;
    /** 归一化方式 */
    norm?: NormType;
    /** Mel 频率转换公式 */
    mel_scale?: MelScaleType;
    /** 迭代次数 (用于伪逆估计) */
    n_iter?: number;
    /** 容差 */
    tolerance_loss?: number;
    /** 容差变化 */
    tolerance_change?: number;
}

/**
 * 逆 Mel 频率尺度变换
 *
 * 从 Mel 频谱图估计普通频谱图 (使用伪逆方法)
 *
 * @example
 * ```ts
 * const inverseMelScale = new InverseMelScale({ n_stft: 201, n_mels: 80 });
 * const specgram = await inverseMelScale.forward(melSpec);
 * ```
 */
export class InverseMelScale extends Module {
    private n_stft: number;
    private n_mels: number;
    private sample_rate: number;
    private f_min: number;
    private f_max: number;
    private norm: NormType;
    private mel_scale: MelScaleType;
    private n_iter: number;
    private fb_pinv!: Tensor;

    constructor(options: InverseMelScaleOptions = {}) {
        super();

        this.n_stft = options.n_stft ?? 201;
        this.n_mels = options.n_mels ?? 128;
        this.sample_rate = options.sample_rate ?? 16000;
        this.f_min = options.f_min ?? 0.0;
        this.f_max = options.f_max ?? this.sample_rate / 2;
        this.norm = options.norm ?? null;
        this.mel_scale = options.mel_scale ?? "htk";
        this.n_iter = options.n_iter ?? 10;

        // 直接创建伪逆滤波器组矩阵
        const fb = melscaleFbanks(
            this.n_stft,
            this.f_min,
            this.f_max,
            this.n_mels,
            this.sample_rate,
            this.norm,
            this.mel_scale
        );
        // 使用伪逆: pinv(fb) = (fb^T @ fb)^-1 @ fb^T
        // 简化为转置近似
        const fb_pinv = k.transpose(fb, 0, 1);
        this.fb_pinv = fb_pinv;
        this.registerBuffer("fb_pinv", fb_pinv);
    }

    /**
     * 从 Mel 频谱图估计普通频谱图
     *
     * @param melspec - Mel 频谱图 (..., n_mels, time)
     * @returns 估计的频谱图 (..., n_stft, time)
     */
    async forward(melspec: Tensor): Promise<Tensor> {
        // melspec: (..., M, T), fb_pinv: (M, F)
        // 输出: (..., F, T)

        const shape = melspec.shape;
        const ndim = shape.length;

        // Transpose: (..., M, T) -> (..., T, M)
        const transposed = k.transpose(melspec, ndim - 2, ndim - 1);

        // matmul: (..., T, M) @ (M, F) -> (..., T, F)
        const result = k.matmul(transposed, this.fb_pinv);

        // Transpose back: (..., T, F) -> (..., F, T)
        return k.transpose(result, ndim - 2, ndim - 1);
    }
}
