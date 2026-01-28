/**
 * MuLawEncoding / MuLawDecoding Transforms
 *
 * 对标 torchaudio.transforms.MuLawEncoding / MuLawDecoding
 *
 * μ-law 压缩编解码转换类
 */

import { type Tensor } from "../../tensor";
import { Module } from "../../nn/module";
import { muLawEncoding as encodeFn, muLawDecoding as decodeFn } from "../functional";

export interface MuLawEncodingOptions {
    /** 量化级数 (默认: 256, 即 8-bit) */
    quantization_channels?: number;
}

/**
 * μ-law 编码转换类
 *
 * 将 [-1, 1] 范围的信号压缩编码为 [0, quantization_channels - 1] 的整数
 *
 * @example
 * ```ts
 * const encoder = new MuLawEncoding({ quantization_channels: 256 });
 * const encoded = await encoder.forward(waveform);
 * ```
 */
export class MuLawEncoding extends Module {
    private quantization_channels: number;

    constructor(options: MuLawEncodingOptions = {}) {
        super();
        this.quantization_channels = options.quantization_channels ?? 256;
    }

    /**
     * 编码信号
     *
     * @param x - 输入信号，范围 [-1, 1]
     * @returns 编码后的信号，范围 [0, quantization_channels - 1]
     */
    async forward(x: Tensor): Promise<Tensor> {
        return encodeFn(x, this.quantization_channels);
    }
}

/**
 * μ-law 解码转换类
 *
 * 将 [0, quantization_channels - 1] 范围的编码信号解码回 [-1, 1]
 *
 * @example
 * ```ts
 * const decoder = new MuLawDecoding({ quantization_channels: 256 });
 * const decoded = await decoder.forward(encoded);
 * ```
 */
export class MuLawDecoding extends Module {
    private quantization_channels: number;

    constructor(options: MuLawEncodingOptions = {}) {
        super();
        this.quantization_channels = options.quantization_channels ?? 256;
    }

    /**
     * 解码信号
     *
     * @param x_mu - 编码信号，范围 [0, quantization_channels - 1]
     * @returns 解码后的信号，范围 [-1, 1]
     */
    async forward(x_mu: Tensor): Promise<Tensor> {
        return decodeFn(x_mu, this.quantization_channels);
    }
}
