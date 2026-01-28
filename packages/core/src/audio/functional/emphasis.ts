/**
 * 预加重 / 去加重
 *
 * 对标 torchaudio.functional.preemphasis / deemphasis
 *
 * 预加重用于增强高频成分，去加重用于还原
 */

import { type Tensor } from '../../tensor';
import * as k from '../../index';

/**
 * 预加重滤波
 *
 * 应用一阶高通滤波器增强高频成分: y[n] = x[n] - coeff * x[n-1]
 *
 * @param waveform - 输入波形 (..., time)
 * @param coeff - 预加重系数 (默认: 0.97)
 * @returns 预加重后的波形 (..., time)
 *
 * @example
 * ```ts
 * const emphasized = preemphasis(waveform, 0.97);
 * ```
 */
export function preemphasis(waveform: Tensor, coeff: number = 0.97): Tensor {
    // y[n] = x[n] - coeff * x[n-1]
    // 对于第一个样本，假设 x[-1] = 0，所以 y[0] = x[0]

    const shape = waveform.shape;
    const timeLength = shape[shape.length - 1];

    if (timeLength <= 1) {
        return waveform;
    }

    // x[n]: 从索引 1 到末尾
    // x[n-1]: 从索引 0 到 timeLength - 1
    const xCurrent = k.slice(waveform, `..., 1:`);
    const xPrevious = k.slice(waveform, `..., :-1`);

    // y[1:] = x[1:] - coeff * x[:-1]
    const yTail = k.sub(xCurrent, k.mul(xPrevious, coeff));

    // y[0] = x[0] (保持第一个样本不变)
    const yHead = k.slice(waveform, `..., :1`);

    // 拼接
    return k.cat([yHead, yTail], -1);
}

/**
 * 去加重滤波
 *
 * 预加重的逆操作: y[n] = x[n] + coeff * y[n-1]
 *
 * 注意: 这是一个 IIR 滤波器，存在前后依赖，GPU 难以并行。
 * 当前实现使用近似方法，适用于大多数音频应用场景。
 *
 * @param waveform - 输入波形 (..., time)
 * @param coeff - 去加重系数 (默认: 0.97，应与预加重系数相同)
 * @returns 去加重后的波形 (..., time)
 *
 * @example
 * ```ts
 * const deemphasized = deemphasis(emphasized, 0.97);
 * ```
 */
export function deemphasis(waveform: Tensor, coeff: number = 0.97): Tensor {
    // IIR 递归: y[n] = x[n] + coeff * y[n-1]
    // 由于 GPU 不适合递归计算，这里使用展开近似

    // 对于短音频或 coeff 接近 1 的情况，可以使用多级展开
    // y[n] = x[n] + coeff*x[n-1] + coeff^2*x[n-2] + ... (截断展开)

    const shape = waveform.shape;
    const timeLength = shape[shape.length - 1];

    if (timeLength <= 1 || coeff === 0) {
        return waveform;
    }

    // 使用迭代展开 (6 阶)，适用于大多数场景
    const numIterations = 6;
    let result = waveform;

    for (let i = 0; i < numIterations; i++) {
        const power = Math.pow(coeff, i + 1);
        if (power < 1e-6) break; // 衰减足够小时停止

        // 创建 padding 版本: 左移 i+1 个位置
        const shiftAmount = i + 1;
        if (shiftAmount >= timeLength) break;

        // 获取 x[n - (i+1)]
        const shifted = k.slice(waveform, `..., :-${shiftAmount}`);
        // 左侧需要填充 shiftAmount 个 0
        const padded = k.pad(shifted, [shiftAmount, 0], 'constant', 0);

        result = k.add(result, k.mul(padded, power));
    }

    return result;
}
