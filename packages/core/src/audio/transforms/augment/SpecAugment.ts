/**
 * SpecAugment Augmentations
 *
 * 对标 torchaudio.transforms.FrequencyMasking 和 TimeMasking
 *
 * 实现 SpecAugment 论文中的数据增强技术
 * @see https://arxiv.org/abs/1904.08779
 */

import { type Tensor } from "../../../tensor";
import { Module } from "../../../nn/module";
import * as k from "../../../index";

export interface FrequencyMaskingOptions {
    /** 最大频率遮罩宽度 */
    freq_mask_param: number;
    /** 是否对每个样本使用独立遮罩 */
    iid_masks?: boolean;
}

/**
 * 频率遮罩变换
 *
 * 在频谱图的频率维度上应用随机遮罩
 *
 * @example
 * ```ts
 * const transform = new FrequencyMasking({ freq_mask_param: 27 });
 * const augmented = await transform.forward(spectrogram);
 * ```
 */
export class FrequencyMasking extends Module {
    private freq_mask_param: number;
    private iid_masks: boolean;

    constructor(options: FrequencyMaskingOptions) {
        super();
        this.freq_mask_param = options.freq_mask_param;
        this.iid_masks = options.iid_masks ?? false;
    }

    /**
     * 应用频率遮罩
     *
     * @param specgram - 输入频谱图 (..., freq, time)
     * @param mask_value - 遮罩填充值 (默认: 0)
     * @returns 应用遮罩后的频谱图
     */
    async forward(specgram: Tensor, mask_value: number = 0): Promise<Tensor> {
        const shape = specgram.shape;
        const ndim = shape.length;
        const freq_dim = ndim - 2;
        const n_freq = shape[freq_dim];

        // 确保遮罩宽度不超过频率数
        const maxWidth = Math.min(this.freq_mask_param, n_freq);

        if (maxWidth === 0) {
            return specgram;
        }

        // 随机选择遮罩宽度和起始位置
        // 注意：这里使用 JavaScript 随机数，实际应用中可能需要更好的随机性
        const width = Math.floor(Math.random() * (maxWidth + 1));
        const start = Math.floor(Math.random() * (n_freq - width + 1));

        if (width === 0) {
            return specgram;
        }

        // 创建遮罩索引
        // 使用切片实现：将 specgram[..., start:start+width, :] 设为 mask_value

        // 由于我们没有原地操作，使用 cat 组合非遮罩和遮罩区域
        // 这会创建一个新张量

        // 分三部分：[..., :start, :], [..., start:start+width, :], [..., start+width:, :]
        const slicePrefix = `${ndim > 2 ? ":," : ""}:${start},:`;
        const sliceMask = `${ndim > 2 ? ":," : ""}${start}:${start + width},:`;
        const sliceSuffix = `${ndim > 2 ? ":," : ""}${start + width}:,:`;

        const prefix = k.slice(specgram, slicePrefix);
        const maskRegion = k.slice(specgram, sliceMask);
        const suffix = k.slice(specgram, sliceSuffix);

        // 创建填充值张量
        const maskedRegion = k.full([...maskRegion.shape], mask_value, specgram.dtype);

        // 沿频率维度 cat
        let result: Tensor;
        if (start === 0) {
            result = k.cat([maskedRegion, suffix], freq_dim);
        } else if (start + width >= n_freq) {
            result = k.cat([prefix, maskedRegion], freq_dim);
        } else {
            result = k.cat([prefix, maskedRegion, suffix], freq_dim);
        }

        return result;
    }
}

export interface TimeMaskingOptions {
    /** 最大时间遮罩宽度 */
    time_mask_param: number;
    /** 是否对每个样本使用独立遮罩 */
    iid_masks?: boolean;
    /** 时间遮罩宽度占比上限 (0-1) */
    p?: number;
}

/**
 * 时间遮罩变换
 *
 * 在频谱图的时间维度上应用随机遮罩
 *
 * @example
 * ```ts
 * const transform = new TimeMasking({ time_mask_param: 100 });
 * const augmented = await transform.forward(spectrogram);
 * ```
 */
export class TimeMasking extends Module {
    private time_mask_param: number;
    private iid_masks: boolean;
    private p: number;

    constructor(options: TimeMaskingOptions) {
        super();
        this.time_mask_param = options.time_mask_param;
        this.iid_masks = options.iid_masks ?? false;
        this.p = options.p ?? 1.0;
    }

    /**
     * 应用时间遮罩
     *
     * @param specgram - 输入频谱图 (..., freq, time)
     * @param mask_value - 遮罩填充值 (默认: 0)
     * @returns 应用遮罩后的频谱图
     */
    async forward(specgram: Tensor, mask_value: number = 0): Promise<Tensor> {
        const shape = specgram.shape;
        const ndim = shape.length;
        const time_dim = ndim - 1;
        const n_time = shape[time_dim];

        // 考虑 p 参数限制遮罩宽度
        const maxWidthByP = Math.floor(n_time * this.p);
        const maxWidth = Math.min(this.time_mask_param, maxWidthByP, n_time);

        if (maxWidth === 0) {
            return specgram;
        }

        // 随机选择遮罩宽度和起始位置
        const width = Math.floor(Math.random() * (maxWidth + 1));
        const start = Math.floor(Math.random() * (n_time - width + 1));

        if (width === 0) {
            return specgram;
        }

        // 创建遮罩 - 沿时间维度切片
        const slicePrefix = `${ndim > 2 ? ":," : ""}:,:${start}`;
        const sliceMask = `${ndim > 2 ? ":," : ""}:,${start}:${start + width}`;
        const sliceSuffix = `${ndim > 2 ? ":," : ""}:,${start + width}:`;

        const prefix = k.slice(specgram, slicePrefix);
        const maskRegion = k.slice(specgram, sliceMask);
        const suffix = k.slice(specgram, sliceSuffix);

        // 创建填充值张量
        const maskedRegion = k.full([...maskRegion.shape], mask_value, specgram.dtype);

        // 沿时间维度 cat
        let result: Tensor;
        if (start === 0) {
            result = k.cat([maskedRegion, suffix], time_dim);
        } else if (start + width >= n_time) {
            result = k.cat([prefix, maskedRegion], time_dim);
        } else {
            result = k.cat([prefix, maskedRegion, suffix], time_dim);
        }

        return result;
    }
}
