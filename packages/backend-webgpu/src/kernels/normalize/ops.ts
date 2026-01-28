/**
 * Normalize Operations Configuration
 * 
 * 所有 Normalize 操作的配置注册
 */

import type { NormalizeOpConfig } from './types';

/**
 * NORMALIZE_OPS - 所有 Normalize 操作的配置
 */
export const NORMALIZE_OPS: Record<string, NormalizeOpConfig> = {
    /**
     * Softmax
     * 
     * softmax(x, dim) = exp(x - max(x, dim)) / sum(exp(x - max(x, dim)), dim)
     * 
     * 数值稳定: 减去 max 防止 exp 溢出
     */
    'softmax': {
        kind: 'softmax',
        statistics: ['max', 'sum_exp'],
    },

    /**
     * Log Softmax
     * 
     * log_softmax(x, dim) = x - max(x, dim) - log(sum(exp(x - max(x, dim)), dim))
     * 
     * 避免 log(softmax(x)) 的数值问题
     */
    'log_softmax': {
        kind: 'log_softmax',
        statistics: ['max', 'sum_exp'],
    },

    /**
     * Softmin
     * 
     * softmin(x, dim) = softmax(-x, dim)
     */
    'softmin': {
        kind: 'softmin',
        statistics: ['max', 'sum_exp'],
        negateInput: true,
    },

    /**
     * Layer Normalization
     * 
     * y = (x - mean) / sqrt(var + eps) * weight + bias
     * 
     * 沿 normalized_shape 维度归一化
     */
    'layer_norm': {
        kind: 'layer_norm',
        statistics: ['mean', 'var'],
        hasAffine: true,
        varianceAlgorithm: 'welford',
    },

    /**
     * Batch Normalization
     * 
     * y = (x - mean) / sqrt(var + eps) * weight + bias
     * 
     * - training=True: 使用 batch 统计量
     * - training=False: 使用 running_mean/running_var
     */
    'batch_norm': {
        kind: 'batch_norm',
        statistics: ['mean', 'var'],
        hasAffine: true,
        hasRunningStats: true,
        varianceAlgorithm: 'welford',
    },

    /**
     * Group Normalization
     * 
     * 将 C 通道分成 num_groups 组，每组独立归一化
     */
    'group_norm': {
        kind: 'group_norm',
        statistics: ['mean', 'var'],
        hasAffine: true,
        varianceAlgorithm: 'welford',
    },

    /**
     * RMS Normalization
     * 
     * y = x / sqrt(mean(x²) + eps) * weight
     * 
     * 无 mean centering，计算更快
     */
    'rms_norm': {
        kind: 'rms_norm',
        statistics: ['rms'],
        hasAffine: true,  // 只有 weight，无 bias
    },

    /**
     * Lp Normalize (F.normalize)
     * 
     * y = x / max(norm(x, p, dim), eps)
     * 
     * 其中 norm(x, p, dim) = (sum(|x|^p, dim))^(1/p)
     */
    'lp_normalize': {
        kind: 'lp_normalize',
        statistics: ['norm'],
    },
};
