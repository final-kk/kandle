/**
 * Scan Operations Registry
 * 
 * 所有扫描操作的 WGSL 表达式定义
 * 参考: PyTorch cumsum, cumprod, cummax, cummin
 */

import { ScanOpConfig } from './types';
import { WGSL_CONSTANTS } from '../../base/dtype';

/**
 * 扫描操作注册表
 * 
 * 每个操作必须:
 * 1. operator: 是结合律的 (associative): (a⊕b)⊕c = a⊕(b⊕c)
 * 2. identity: 有恒等元 e 使得 e⊕a = a⊕e = a
 */
export const SCAN_OPS: Record<string, ScanOpConfig> = {
    // ================================================================
    // 累积求和
    // ================================================================

    'cumsum': {
        operator: (a, b) => `(${a} + ${b})`,
        identity: (t) => `${t}(0)`,
        hasIndices: false,
    },

    // ================================================================
    // 累积求积
    // ================================================================

    'cumprod': {
        operator: (a, b) => `(${a} * ${b})`,
        identity: (t) => `${t}(1)`,
        hasIndices: false,
    },

    // ================================================================
    // 累积最大值 (带索引)
    // ================================================================

    'cummax': {
        operator: (a, b) => `max(${a}, ${b})`,
        identity: (t) => {
            if (t === 'f32' || t === 'f16') {
                return `${t}(${WGSL_CONSTANTS.NEG_FLT_MAX})`;
            } else if (t === 'i32') {
                return `i32(${WGSL_CONSTANTS.INT_MIN})`;
            } else if (t === 'u32') {
                return 'u32(0)';
            }
            return `${t}(0)`;
        },
        hasIndices: true,
        compare: (newVal, curVal) => `${newVal} > ${curVal}`,
    },

    // ================================================================
    // 累积最小值 (带索引)
    // ================================================================

    'cummin': {
        operator: (a, b) => `min(${a}, ${b})`,
        identity: (t) => {
            if (t === 'f32' || t === 'f16') {
                return `${t}(${WGSL_CONSTANTS.FLT_MAX})`;
            } else if (t === 'i32') {
                return `i32(${WGSL_CONSTANTS.INT_MAX})`;
            } else if (t === 'u32') {
                return `u32(${WGSL_CONSTANTS.UINT_MAX})`;
            }
            return `${t}(0)`;
        },
        hasIndices: true,
        compare: (newVal, curVal) => `${newVal} < ${curVal}`,
    },
};
