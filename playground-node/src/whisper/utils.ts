/**
 * Whisper 工具函数和配置
 *
 * 包含：
 * - 模型配置选择
 * - 日志工具
 * - 张量统计函数
 */

import { Tensor } from '@kandle/core';
import {
    type WhisperModelSize,
    type WhisperConfig,
    getWhisperConfig,
    WHISPER_SPECIAL_TOKENS,
} from '@kandle/model-utils';

// ============================================================================
// 默认配置
// ============================================================================

/** 默认模型尺寸 */
export const DEFAULT_MODEL_SIZE: WhisperModelSize = 'base';

/** EOT (End of Transcript) token ID */
export const EOT_TOKEN_ID = WHISPER_SPECIAL_TOKENS.EOT;

// ============================================================================
// 日志工具
// ============================================================================

export const logger = {
    log: console.log,
    logGroup: (msg: string) => console.log(`\n=== ${msg} ===`),
    info: console.log,
    success: (msg: string) => console.log(`\x1b[32m${msg}\x1b[0m`), // 绿色
    error: console.error,
    warn: console.warn,
};

// ============================================================================
// 工具函数
// ============================================================================

/** 断言函数 */
export function assert(condition: boolean, msg: string): void {
    if (!condition) {
        logger.error(`断言失败: ${msg}`);
        throw new Error(msg);
    }
}

/** 计算张量的统计信息（最小值、最大值、均值、是否有限） */
export async function getTensorStats(tensor: Tensor): Promise<{
    min: number;
    max: number;
    mean: number;
    isFinite: boolean;
}> {
    const data = await tensor.dataAsync();
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let allFinite = true;

    for (let i = 0; i < data.length; i++) {
        const val = Number(data[i]);
        if (!Number.isFinite(val)) {
            allFinite = false;
        }
        if (val < min) min = val;
        if (val > max) max = val;
        sum += val;
    }

    return {
        min,
        max,
        mean: sum / data.length,
        isFinite: allFinite,
    };
}
