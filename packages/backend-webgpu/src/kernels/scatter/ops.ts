/**
 * Scatter Kernels - Operations Registry
 *
 * 注册所有 Scatter 操作的配置
 */

import type { ScatterOp, ScatterOpConfig, ScatterReduceMode } from './types';

/**
 * Scatter 操作配置表
 */
export const SCATTER_OPS: Record<ScatterOp, ScatterOpConfig> = {
    'scatter': {
        name: 'scatter',
        needsAtomic: false,
    },
    'scatter_add': {
        name: 'scatter_add',
        needsAtomic: true,
        reduceMode: 'sum',
    },
    'scatter_reduce': {
        name: 'scatter_reduce',
        needsAtomic: true,
        // reduceMode 和 includeSelf 由外部参数传入
    },
};

/**
 * 根据归约模式获取操作配置
 */
export function getScatterReduceConfig(
    reduce: ScatterReduceMode,
    includeSelf: boolean
): ScatterOpConfig {
    return {
        name: 'scatter_reduce',
        needsAtomic: true,
        reduceMode: reduce,
        includeSelf,
    };
}

/**
 * 获取归约操作的初始值 (WGSL 格式)
 */
export function getReduceIdentity(
    reduce: ScatterReduceMode,
    dtype: string,
    wgslType: string
): string {
    switch (reduce) {
        case 'sum':
            return `${wgslType}(0)`;
        case 'prod':
            return `${wgslType}(1)`;
        case 'mean':
            return `${wgslType}(0)`;  // 累加，最后除以计数
        case 'amax':
            // 负无穷
            if (wgslType === 'f32') return 'f32(-3.402823e+38)';
            if (wgslType === 'i32') return 'i32(-2147483647)';
            return 'u32(0)';
        case 'amin':
            // 正无穷
            if (wgslType === 'f32') return 'f32(3.402823e+38)';
            if (wgslType === 'i32') return 'i32(2147483647)';
            return 'u32(4294967295)';
    }
}

/**
 * 获取归约操作的 WGSL 合并表达式
 */
export function getReduceOp(
    reduce: ScatterReduceMode,
    oldVar: string,
    newVar: string
): string {
    switch (reduce) {
        case 'sum':
        case 'mean':
            return `${oldVar} + ${newVar}`;
        case 'prod':
            return `${oldVar} * ${newVar}`;
        case 'amax':
            return `max(${oldVar}, ${newVar})`;
        case 'amin':
            return `min(${oldVar}, ${newVar})`;
    }
}
