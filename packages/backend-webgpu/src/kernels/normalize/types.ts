/**
 * Normalize Kernel Types
 * 
 * 类型定义: softmax, log_softmax, layer_norm, batch_norm, group_norm, rms_norm, normalize
 */

import type { ITensorHandle } from '@kandle/types';

/**
 * Normalize 操作类型
 */
export type NormalizeKind =
    | 'softmax'
    | 'log_softmax'
    | 'softmin'
    | 'layer_norm'
    | 'batch_norm'
    | 'group_norm'
    | 'rms_norm'
    | 'lp_normalize';

/**
 * 统计量类型
 */
export type StatisticType = 'max' | 'sum' | 'sum_exp' | 'mean' | 'var' | 'rms' | 'norm';

/**
 * Normalize 操作配置
 */
export interface NormalizeOpConfig {
    /** 操作类型 */
    kind: NormalizeKind;

    /**
     * 需要计算的统计量
     * - softmax/log_softmax: ['max', 'sum_exp']
     * - layer_norm/batch_norm/group_norm: ['mean', 'var']
     * - rms_norm: ['rms']
     * - lp_normalize: ['norm']
     */
    statistics: StatisticType[];

    /** 是否有仿射参数 (weight, bias) */
    hasAffine?: boolean;

    /** 是否使用 running stats (batch_norm) */
    hasRunningStats?: boolean;

    /**
     * 方差计算算法
     * - 'naive': 简单方差 mean((x - μ)²)
     * - 'welford': 在线单 pass 算法，数值更稳定
     */
    varianceAlgorithm?: 'naive' | 'welford';

    /** 是否在计算前对输入取负 (softmin) */
    negateInput?: boolean;
}

/**
 * Normalize Kernel 参数
 */
export interface NormalizeKernelParams {
    /** 归一化维度 */
    dim?: number | number[];

    /** 数值稳定项 */
    eps?: number;

    /** Lp 范数的 p 值 */
    p?: number;

    /** Group Norm 的组数 */
    numGroups?: number;

    /** Layer Norm 的 normalized_shape */
    normalizedShape?: number[];

    /** 权重 tensor */
    weight?: ITensorHandle;

    /** 偏置 tensor */
    bias?: ITensorHandle;

    /** Running mean (batch_norm) */
    runningMean?: ITensorHandle;

    /** Running var (batch_norm) */
    runningVar?: ITensorHandle;

    /** 训练模式 (batch_norm) */
    training?: boolean;

    /** 输出 tensor */
    out?: ITensorHandle;
}

/**
 * Shader 构建参数
 * 
 * 工业级实现：原生支持 strided (非连续) 输入
 */
export interface NormalizeShaderParams {
    /** 操作配置 */
    config: NormalizeOpConfig;

    /** 输入形状 */
    inputShape: readonly number[];

    /** 输入 strides (用于 strided 访问) */
    inputStrides: readonly number[];

    /** 输入偏移量 */
    inputOffset: number;

    /** 归一化维度 (规范化后的正值) */
    normalizedDims: number[];

    /** 归一化维度的总元素数 */
    reduceSize: number;

    /** 是否有 weight */
    hasWeight: boolean;

    /** 是否有 bias */
    hasBias: boolean;

    /** eps 值 */
    eps: number;

    /** p 值 (for lp_normalize) */
    p?: number;

    /** 组数 (for group_norm) */
    numGroups?: number;

    /** 数据类型 */
    dtype: string;
}
