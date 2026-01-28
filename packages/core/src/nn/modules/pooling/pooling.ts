/**
 * Pooling Modules - 池化层
 */

import { Tensor } from '../../../tensor';
import { Module } from '../../module';
import { functional } from '../../../generated/ops';

/**
 * MaxPool2d
 */
export class MaxPool2d extends Module {
    kernelSize: number | [number, number];
    stride: number | [number, number];
    padding: number | [number, number];
    dilation: number | [number, number];
    ceilMode: boolean;
    returnIndices: boolean;

    constructor(
        kernelSize: number | [number, number],
        stride?: number | [number, number],
        padding: number | [number, number] = 0,
        dilation: number | [number, number] = 1,
        returnIndices: boolean = false,
        ceilMode: boolean = false
    ) {
        super();
        this.kernelSize = kernelSize;
        this.stride = stride ?? kernelSize;
        this.padding = padding;
        this.dilation = dilation;
        this.returnIndices = returnIndices;
        this.ceilMode = ceilMode;
    }

    async forward(input: Tensor): Promise<Tensor> {
        return functional.maxPool2d(
            input,
            this.kernelSize,
            this.stride,
            this.padding,
            this.dilation,
            this.ceilMode,
            this.returnIndices
        );
    }
}

/**
 * AvgPool2d
 */
export class AvgPool2d extends Module {
    kernelSize: number | [number, number];
    stride: number | [number, number];
    padding: number | [number, number];
    ceilMode: boolean;
    countIncludePad: boolean;
    divisorOverride?: number;

    constructor(
        kernelSize: number | [number, number],
        stride?: number | [number, number],
        padding: number | [number, number] = 0,
        ceilMode: boolean = false,
        countIncludePad: boolean = true,
        divisorOverride?: number
    ) {
        super();
        this.kernelSize = kernelSize;
        this.stride = stride ?? kernelSize;
        this.padding = padding;
        this.ceilMode = ceilMode;
        this.countIncludePad = countIncludePad;
        this.divisorOverride = divisorOverride;
    }

    async forward(input: Tensor): Promise<Tensor> {
        return functional.avgPool2d(
            input,
            this.kernelSize,
            this.stride,
            this.padding,
            this.ceilMode,
            this.countIncludePad,
            this.divisorOverride
        );
    }
}

/**
 * MaxPool1d
 */
export class MaxPool1d extends Module {
    kernelSize: number;
    stride: number;
    padding: number;
    dilation: number;
    ceilMode: boolean;
    returnIndices: boolean;

    constructor(
        kernelSize: number,
        stride?: number,
        padding: number = 0,
        dilation: number = 1,
        returnIndices: boolean = false,
        ceilMode: boolean = false
    ) {
        super();
        this.kernelSize = kernelSize;
        this.stride = stride ?? kernelSize;
        this.padding = padding;
        this.dilation = dilation;
        this.returnIndices = returnIndices;
        this.ceilMode = ceilMode;
    }

    async forward(input: Tensor): Promise<Tensor> {
        return functional.maxPool1d(
            input,
            this.kernelSize,
            this.stride,
            this.padding,
            this.dilation,
            this.ceilMode,
            this.returnIndices
        );
    }
}

/**
 * AvgPool1d
 */
export class AvgPool1d extends Module {
    kernelSize: number;
    stride: number;
    padding: number;
    ceilMode: boolean;
    countIncludePad: boolean;

    constructor(
        kernelSize: number,
        stride?: number,
        padding: number = 0,
        ceilMode: boolean = false,
        countIncludePad: boolean = true
    ) {
        super();
        this.kernelSize = kernelSize;
        this.stride = stride ?? kernelSize;
        this.padding = padding;
        this.ceilMode = ceilMode;
        this.countIncludePad = countIncludePad;
    }

    async forward(input: Tensor): Promise<Tensor> {
        return functional.avgPool1d(
            input,
            this.kernelSize,
            this.stride,
            this.padding,
            this.ceilMode,
            this.countIncludePad
        );
    }
}
