/**
 * IIR 滤波器 (Linear Filter)
 *
 * 对标 torchaudio.functional.lfilter
 *
 * v3: 通过 dispatch 机制调用 IIR kernel，保持同步 API
 */

import { Tensor } from '../../tensor';
import * as k from '../../index';
import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const lfilterBiquadEntry = opschema.ops.lfilterBiquad;
import { dispatchIIR, type OperatorContext } from '../../dispatch/handlers';

/**
 * lfilter 选项
 */
export interface LfilterOptions {
    /**
     * 是否将输出限制在 [-1, 1] 范围内
     * @default true
     */
    clamp?: boolean;
}

/**
 * IIR 滤波器 - 通过差分方程实现
 *
 * 实现差分方程:
 * ```
 * a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
 *                       - a[1]*y[n-1] - ... - a[N]*y[n-N]
 * ```
 *
 * 实现策略:
 * - 对于纯 FIR (a=[1]): 使用 conv1d
 * - 对于 IIR: 使用迭代展开法 (通用实现)
 *
 * 注: biquad 滤波器请直接使用 biquad() 函数获得优化的 GPU kernel
 *
 * @param waveform - 输入波形 (..., time)，值应在 [-1, 1] 范围内
 * @param aCoeffs - 分母系数 (num_order + 1)，低延迟系数在前，如 [a0, a1, a2, ...]
 * @param bCoeffs - 分子系数 (num_order + 1)，低延迟系数在前，如 [b0, b1, b2, ...]
 * @param options - 可选参数
 * @returns 滤波后的波形 (..., time)
 */
export function lfilter(
    waveform: Tensor,
    aCoeffs: Tensor,
    bCoeffs: Tensor,
    options: LfilterOptions = {}
): Tensor {
    const { clamp = true } = options;

    // 验证系数
    if (aCoeffs.shape.length !== 1 || bCoeffs.shape.length !== 1) {
        throw new Error('lfilter: aCoeffs and bCoeffs must be 1D tensors');
    }

    if (aCoeffs.numel !== bCoeffs.numel) {
        throw new Error(
            `lfilter: aCoeffs and bCoeffs must have same size. ` +
            `Got aCoeffs: ${aCoeffs.numel}, bCoeffs: ${bCoeffs.numel}`
        );
    }

    const order = aCoeffs.numel;
    if (order === 0) {
        throw new Error('lfilter: coefficients cannot be empty');
    }

    // 标准化系数 (除以 a[0])
    const a0 = k.slice(aCoeffs, '0:1');
    const aNorm = k.div(aCoeffs, a0); // [1, a1/a0, a2/a0, ...]
    const bNorm = k.div(bCoeffs, a0); // [b0/a0, b1/a0, b2/a0, ...]

    // 获取输入形状
    const inputShape = waveform.shape;
    const ndim = inputShape.length;
    const timeLen = inputShape[ndim - 1];

    // 检查是否有 IIR 反馈 (a[1:] 不为零)
    const hasIIR = order > 1;

    // 使用通用实现
    return executeGenericLfilter(waveform, aNorm, bNorm, order, hasIIR, clamp, inputShape, timeLen);
}

/**
 * Biquad IIR 滤波器 (优化版本)
 *
 * 接收标量系数，直接调度到 GPU kernel
 *
 * 差分方程: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
 *
 * @param waveform - 输入波形 (..., time)
 * @param b0 - 分子系数 x[n]
 * @param b1 - 分子系数 x[n-1]
 * @param b2 - 分子系数 x[n-2]
 * @param a1 - 分母系数 y[n-1] (已归一化，a0=1)
 * @param a2 - 分母系数 y[n-2] (已归一化，a0=1)
 * @param options - 可选参数
 * @returns 滤波后的波形 (..., time)
 */
export function lfilterBiquad(
    waveform: Tensor,
    b0: number,
    b1: number,
    b2: number,
    a1: number,
    a2: number,
    options: LfilterOptions = {}
): Tensor {
    const { clamp = true } = options;

    // 确保输入是连续的 (IIR kernel 需要连续内存)
    const contiguousWaveform = waveform.contiguous();

    // 构造 OperatorContext
    const ctx: OperatorContext = {
        opName: 'iir.biquad',
        tensorInputs: [contiguousWaveform._handle],
        scalarArgs: { b0, b1, b2, a1, a2, clamp },
        metadata: {},
    };

    // 调用 IIR dispatch
    const resultHandle = dispatchIIR(lfilterBiquadEntry, ctx);

    // 包装为 Tensor
    return new Tensor(resultHandle as ITensorHandle);
}

/**
 * 通用 lfilter 实现 (使用 tensor 操作)
 */
function executeGenericLfilter(
    waveform: Tensor,
    aNorm: Tensor,
    bNorm: Tensor,
    order: number,
    hasIIR: boolean,
    clampOutput: boolean,
    inputShape: readonly number[],
    timeLen: number
): Tensor {
    // 保存原始 batch 维度，将输入 reshape 为 (batch, 1, 1, time) 用于 conv2d
    const batchDims = inputShape.slice(0, -1);
    const batchSize = batchDims.reduce((a, b) => a * b, 1);

    // Reshape: (..., time) -> (batch, 1, 1, time) 用于 conv2d
    let input4d = k.reshape(waveform, [batchSize, 1, 1, timeLen]);

    // ========================================================================
    // FIR 部分: 使用 conv2d 实现 sum(b[i] * x[n-i])
    // ========================================================================

    // bCoeffs 需要 reshape 为 (out=1, in=1, kH=1, kW=order)
    // 使用 flip 实现翻转: bFlipped = flip(bNorm, 0)
    const bFlipped = k.flip(bNorm, [0]);
    const bKernel = k.reshape(bFlipped, [1, 1, 1, order]);

    // 左侧 padding 保持因果性: pad (order-1) 在最后一维左边
    const paddedInput = k.pad(input4d, [order - 1, 0, 0, 0], 'constant', 0);

    // conv2d: (batch, 1, 1, time + order - 1) * (1, 1, 1, order) -> (batch, 1, 1, time)
    let result = k.conv2d(paddedInput, bKernel);

    // ========================================================================
    // IIR 部分: 迭代展开 IIR 反馈
    // ========================================================================

    if (hasIIR) {
        // 获取 IIR 系数 a[1], a[2], ... (归一化后)
        const aIIR = k.slice(aNorm, '1:');

        const maxIter = Math.min(timeLen, 64);

        for (let iter = 0; iter < maxIter; iter++) {
            let iirCorrection = k.zerosLike(result);

            // 对每个 IIR 系数应用延迟
            for (let i = 0; i < order - 1; i++) {
                const delay = i + 1;
                // a_coeff = aIIR[i]
                const aCoeff = k.slice(aIIR, `${i}:${i + 1}`);

                // shift(result, delay): 左移 delay 个位置，右边补零
                if (delay < timeLen) {
                    const shifted = k.slice(result, `..., :-${delay}`);
                    // pad 最后一维 (W 维度)，H 维度不 pad
                    const padded = k.pad(shifted, [delay, 0, 0, 0], 'constant', 0);
                    iirCorrection = k.add(iirCorrection, k.mul(padded, aCoeff));
                }
            }

            // result = FIR_output - iirCorrection
            result = k.sub(k.conv2d(paddedInput, bKernel), iirCorrection);
        }
    }

    // Reshape 回原始形状
    result = k.reshape(result, [...inputShape]);

    // 可选 clamp
    if (clampOutput) {
        result = k.clamp(result, -1.0, 1.0);
    }

    return result;
}
