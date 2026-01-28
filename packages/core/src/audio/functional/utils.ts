/**
 * Audio Utility Functions
 *
 * 对标 torchaudio.functional 中的工具函数：
 * - amplitude_to_DB: 振幅/功率到分贝转换
 * - DB_to_amplitude: 分贝到振幅/功率转换
 */

import { type Tensor } from '../../tensor';
import * as k from '../../index';

/**
 * 将振幅/功率谱转换为分贝刻度
 *
 * @param x - 输入张量 (振幅或功率谱)
 * @param multiplier - 乘数 (10 for power, 20 for amplitude)
 * @param amin - 最小值下限，防止 log(0)
 * @param db_multiplier - 分贝乘数 (通常为 1.0)
 * @param top_db - 可选，动态范围压缩上限
 * @returns 分贝刻度的张量
 *
 * @example
 * ```ts
 * // 功率谱到分贝
 * const db = amplitudeToDB(powerSpec, 10.0);
 * // 振幅谱到分贝
 * const db = amplitudeToDB(ampSpec, 20.0);
 * ```
 */
export function amplitudeToDB(
    x: Tensor,
    multiplier: number,
    amin: number = 1e-10,
    db_multiplier: number = 1.0,
    top_db?: number
): Tensor {
    // 公式: multiplier * log10(clamp(x, min=amin)) * db_multiplier

    // Step 1: Clamp to prevent log(0)
    const clamped = k.clamp(x, amin);

    // Step 2: log10
    const logVal = k.log10(clamped);

    // Step 3: Scale by multiplier and db_multiplier
    let db = k.mul(logVal, multiplier * db_multiplier);

    // Step 4: Optional top_db clipping
    if (top_db !== undefined) {
        // db_max = max(db)
        // db = clamp(db, min=db_max - top_db)
        const dbMax = k.max(db);
        const minDb = k.sub(dbMax, top_db);
        db = k.maximum(db, minDb);
    }

    return db;
}

/**
 * 将分贝刻度转换回振幅/功率
 *
 * @param x - 输入张量 (分贝刻度)
 * @param ref - 参考值
 * @param power - 功率指数 (1 for amplitude, 0.5 for power)
 * @returns 线性刻度的张量
 *
 * @example
 * ```ts
 * // 分贝到功率
 * const power = DBToAmplitude(db, 1.0, 1.0);
 * // 分贝到振幅
 * const amp = DBToAmplitude(db, 1.0, 0.5);
 * ```
 */
export function DBToAmplitude(
    x: Tensor,
    ref: number,
    power: number
): Tensor {
    // 公式: ref * 10^(x * 0.1) ^ power
    // 等价于: ref * 10^(x * 0.1 * power)
    // power=1 for power spectrogram, power=0.5 for amplitude spectrogram

    const exponent = k.mul(x, 0.1 * power);
    const powerOf10 = k.pow(k.full([...x.shape], 10, x.dtype), exponent);
    return k.mul(powerOf10, ref);
}
