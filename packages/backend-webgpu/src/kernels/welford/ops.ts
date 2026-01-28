/**
 * Welford Operations Registry
 * 
 * 注册 variance 和 std 操作
 */

import type { WelfordOpConfig } from './types';

/**
 * Welford 操作注册表
 * 
 * dispatchKey -> 配置
 */
export const WELFORD_OPS: Record<string, WelfordOpConfig> = {
    /**
     * variance: 方差
     * 公式: m2 / (n - correction)
     */
    'variance': {
        applySqrt: false,
    },

    /**
     * std: 标准差
     * 公式: sqrt(m2 / (n - correction))
     */
    'std': {
        applySqrt: true,
    },
};
