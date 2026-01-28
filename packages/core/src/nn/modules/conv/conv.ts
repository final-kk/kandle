/**
 * Conv - 卷积层
 *
 * @see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
 */

import type { DType } from '@kandle/types';
import { Tensor } from '../../../tensor';
import { Module } from '../../module';
import { Parameter } from '../../parameter';
import { functional, empty, zeros } from '../../../generated/ops';

/**
 * 卷积层通用选项
 */
export interface ConvOptions {
    /** 输入通道数 */
    inChannels: number;
    /** 输出通道数 */
    outChannels: number;
    /** 卷积核大小 */
    kernelSize: number | number[];
    /** 步长 (默认 1) */
    stride?: number | number[];
    /** 填充 (默认 0) */
    padding?: number | number[] | 'valid' | 'same';
    /** 膨胀率 (默认 1) */
    dilation?: number | number[];
    /** 分组数 (默认 1) */
    groups?: number;
    /** 是否使用偏置 (默认 true) */
    bias?: boolean;
    /** 填充模式 (默认 'zeros'，暂未使用) */
    paddingMode?: string;
    /** 数据类型 */
    dtype?: DType;
}

/**
 * 卷积层基类
 */
export abstract class _ConvNd extends Module {
    readonly inChannels: number;
    readonly outChannels: number;
    readonly kernelSize: number | number[];
    readonly stride: number | number[];
    readonly padding: number | number[] | 'valid' | 'same';
    readonly dilation: number | number[];
    readonly groups: number;
    readonly paddingMode: string;

    weight: Parameter;
    bias: Parameter | null;

    constructor(
        inChannels: number,
        outChannels: number,
        kernelSize: number | number[],
        stride: number | number[],
        padding: number | number[] | 'valid' | 'same',
        dilation: number | number[],
        groups: number,
        bias: boolean,
        paddingMode: string,
        dtype: DType = 'float32'
    ) {
        super();

        if (inChannels % groups !== 0) {
            throw new Error('in_channels must be divisible by groups');
        }
        if (outChannels % groups !== 0) {
            throw new Error('out_channels must be divisible by groups');
        }

        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
        this.dilation = dilation;
        this.groups = groups;
        this.paddingMode = paddingMode;

        // 计算权重形状
        // Transposed convolution has different weight shape, but here we assume normal conv
        // Weight shape: [out_channels, in_channels / groups, ...kernel_size]
        const weightShape = [outChannels, inChannels / groups, ...(Array.isArray(kernelSize) ? kernelSize : [kernelSize])];

        this.weight = new Parameter(empty(weightShape, dtype));
        this.registerParameter('weight', this.weight);

        if (bias) {
            this.bias = new Parameter(zeros([outChannels], dtype));
            this.registerParameter('bias', this.bias);
        } else {
            this.bias = null;
            this.registerParameter('bias', null);
        }

        this.resetParameters();
    }

    resetParameters(): void {
        // TODO: Implement kaiming_uniform_
        // For now, leave uninitialized as per Linear module precedent
    }

    extraRepr(): string {
        const s = `in_channels=${this.inChannels}, out_channels=${this.outChannels}, kernel_size=${JSON.stringify(this.kernelSize)}, stride=${JSON.stringify(this.stride)}`;
        if (this.padding !== 0) {
            return s + `, padding=${JSON.stringify(this.padding)}`;
        }
        if (this.dilation !== 1) { // generic check, actual check might be array
            return s + `, dilation=${JSON.stringify(this.dilation)}`;
        }
        if (this.groups !== 1) {
            return s + `, groups=${this.groups}`;
        }
        if (this.bias === null) {
            return s + `, bias=false`;
        }
        return s;
    }
}

/**
 * Conv1d - 一维卷积层
 */
export class Conv1d extends _ConvNd {
    constructor(
        inChannels: number,
        outChannels: number,
        kernelSize: number | number[],
        stride: number | number[] = 1,
        padding: number | number[] | 'valid' | 'same' = 0,
        dilation: number | number[] = 1,
        groups: number = 1,
        bias: boolean = true,
        paddingMode: string = 'zeros',
        dtype: DType = 'float32'
    ) {
        const k = typeof kernelSize === 'number' ? [kernelSize] : kernelSize;
        const s = typeof stride === 'number' ? [stride] : stride;
        const p = (typeof padding === 'number') ? [padding] : padding;
        const d = typeof dilation === 'number' ? [dilation] : dilation;

        super(inChannels, outChannels, k, s, p, d, groups, bias, paddingMode, dtype);
    }

    async forward(input: Tensor): Promise<Tensor> {
        return functional.conv1d(
            input,
            this.weight,
            this.bias ?? undefined,
            this.stride as number, // OpSchema usually accepts scalar or list, but check strict typing if needed
            this.padding as any,
            this.dilation as number,
            this.groups
        );
    }
}

/**
 * Conv2d - 二维卷积层
 */
export class Conv2d extends _ConvNd {
    constructor(
        inChannels: number,
        outChannels: number,
        kernelSize: number | [number, number],
        stride: number | [number, number] = 1,
        padding: number | [number, number] | 'valid' | 'same' = 0,
        dilation: number | [number, number] = 1,
        groups: number = 1,
        bias: boolean = true,
        paddingMode: string = 'zeros',
        dtype: DType = 'float32'
    ) {
        // Expand scalars to [x, x]
        const k = typeof kernelSize === 'number' ? [kernelSize, kernelSize] : kernelSize;
        const s = typeof stride === 'number' ? [stride, stride] : stride;
        const p = (typeof padding === 'number') ? [padding, padding] : padding; // 'valid'/'same' parsed in dispatcher, but here for base class storage
        const d = typeof dilation === 'number' ? [dilation, dilation] : dilation;

        super(inChannels, outChannels, k, s, p, d, groups, bias, paddingMode, dtype);
    }

    async forward(input: Tensor): Promise<Tensor> {
        return functional.conv2d(
            input,
            this.weight,
            this.bias ?? undefined,
            this.stride,
            this.padding,
            this.dilation,
            this.groups
        );
    }
}
