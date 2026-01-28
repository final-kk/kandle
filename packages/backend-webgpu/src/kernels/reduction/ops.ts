/**
 * Reduction Operations Registry
 * 
 * 所有归约操作的 WGSL 表达式定义
 * 参考: PyTorch ATen/native/ReductionType.h
 * 
 * 复数支持:
 * - sum/mean: 分量独立累加 (vec2<f32> += vec2<f32>)
 * - prod: 需要复数乘法 cmul(a, b)
 * - max/min: 复数无自然顺序，抛出错误
 */

import { ReductionOpConfig } from './types';
import { WGSL_CONSTANTS } from '../../base/dtype';

/**
 * 复数乘法 WGSL 辅助函数生成
 * cmul(a, b) = (a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x)
 */
function complexMulExpr(a: string, b: string): string {
    return `vec2<f32>(${a}.x * ${b}.x - ${a}.y * ${b}.y, ${a}.x * ${b}.y + ${a}.y * ${b}.x)`;
}

/**
 * 判断类型是否是复数 (vec2<f32>)
 */
function isComplexType(t: string): boolean {
    return t === 'vec2<f32>' || t === 'vec2<f64>';
}

export const REDUCTION_OPS: Record<string, ReductionOpConfig> = {
    // ================================================================
    // 求和归约 (支持复数: 分量独立累加)
    // ================================================================

    'sum': {
        initializer: (t) => {
            // 复数初始化为 vec2(0, 0)
            if (isComplexType(t)) {
                return `${t}(0.0, 0.0)`;
            }
            return `${t}(0)`;
        },
        accumulator: (acc, val) => `${acc} = ${acc} + ${val};`,
    },

    // ================================================================
    // 均值归约 (支持复数: 分量独立累加后除以元素数)
    // ================================================================

    'mean': {
        initializer: (t) => {
            if (isComplexType(t)) {
                return `${t}(0.0, 0.0)`;
            }
            return `${t}(0)`;
        },
        accumulator: (acc, val) => `${acc} = ${acc} + ${val};`,
        finalizer: (acc, totalNumel, t) => {
            if (isComplexType(t)) {
                // 复数除法: 除以实数 n 等价于分量各除以 n
                return `${acc} / ${t}(f32(${totalNumel}), 0.0)`;
            }
            return `${acc} / ${t}(${totalNumel})`;
        },
    },

    // ================================================================
    // 最大值归约 (不支持复数: 复数无自然顺序)
    // ================================================================

    'max': {
        initializer: (t) => {
            if (isComplexType(t)) {
                // 不应该到达这里 - dispatcher 应该在之前抛出错误
                throw new Error('max operation is not supported for complex types (no natural ordering)');
            }
            // 使用类型对应的最小值作为初始值
            if (t === 'f32' || t === 'f16') {
                return `${t}(${WGSL_CONSTANTS.NEG_FLT_MAX})`;
            } else if (t === 'i32') {
                return `i32(${WGSL_CONSTANTS.INT_MIN})`;
            } else if (t === 'u32') {
                return 'u32(0)'; // UINT32_MIN
            }
            return `${t}(0)`;
        },
        accumulator: (acc, val) => `${acc} = max(${acc}, ${val});`,
    },

    // ================================================================
    // 最小值归约 (不支持复数: 复数无自然顺序)
    // ================================================================

    'min': {
        initializer: (t) => {
            if (isComplexType(t)) {
                // 不应该到达这里 - dispatcher 应该在之前抛出错误
                throw new Error('min operation is not supported for complex types (no natural ordering)');
            }
            // 使用类型对应的最大值作为初始值
            if (t === 'f32' || t === 'f16') {
                return `${t}(${WGSL_CONSTANTS.FLT_MAX})`;
            } else if (t === 'i32') {
                return `i32(${WGSL_CONSTANTS.INT_MAX})`;
            } else if (t === 'u32') {
                return `u32(${WGSL_CONSTANTS.UINT_MAX})`;
            }
            return `${t}(0)`;
        },
        accumulator: (acc, val) => `${acc} = min(${acc}, ${val});`,
    },

    // ================================================================
    // 乘积归约 (支持复数: 使用复数乘法)
    // ================================================================

    'prod': {
        initializer: (t) => {
            if (isComplexType(t)) {
                // 复数乘法单位元是 (1, 0)
                return `${t}(1.0, 0.0)`;
            }
            return `${t}(1)`;
        },
        accumulator: (acc, val, t) => {
            if (isComplexType(t!)) {
                // 复数乘法: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                return `${acc} = ${complexMulExpr(acc, val)};`;
            }
            return `${acc} = ${acc} * ${val};`;
        },
    },

    // ================================================================
    // 逻辑归约 (不支持复数)
    // ================================================================

    'all': {
        initializer: () => 'true',
        accumulator: (acc, val) => `${acc} = ${acc} && ${val};`,
    },

    'any': {
        initializer: () => 'false',
        accumulator: (acc, val) => `${acc} = ${acc} || ${val};`,
    },

    // ================================================================
    // LogSumExp (Special Handling in Executor, 不支持复数)
    // ================================================================

    'logsumexp': {
        initializer: (t) => `${t}(0)`, // Placeholder
        accumulator: (acc, val) => ``, // Placeholder
    },

    // ================================================================
    // NaN Handling Operations (不支持复数)
    // ================================================================

    'nansum': {
        initializer: (t) => `${t}(0)`,
        accumulator: (acc, val, t) => {
            // For floats, check isNan. For ints, simple sum.
            if (t === 'f32' || t === 'f16') {
                // Use (val != val) check. IMPORTANT: Use ${val} interpolation!
                return `if (!(${val} != ${val})) { ${acc} = ${acc} + ${val}; }`;
            }
            return `${acc} = ${acc} + ${val};`;
        }
    },

    'nanmean': {
        initializer: (t) => `${t}(0)`, // Placeholder for dedicated executor
        accumulator: (acc, val) => ``, // Placeholder
    },
};

