/**
 * Welford Kernel Types
 * 
 * Welford 算法用于数值稳定的方差/标准差计算
 * 
 * 核心思想：在线算法，逐个元素更新 (mean, m2, n) 三元组
 * 合并时使用 parallel combine 避免数值精度问题
 */

/**
 * Welford 操作配置
 */
export interface WelfordOpConfig {
    /**
     * 是否对结果开方
     * - true: std (标准差)
     * - false: variance (方差)
     */
    applySqrt: boolean;
}

/**
 * Welford Uniform Buffer 参数 (Dimensional Reduction)
 */
export interface WelfordDimParams {
    /** 输出元素总数 (parallel loop) */
    outputNumel: number;
    /** 每个输出要归约的元素数 (reduction loop) */
    reductionNumel: number;
    /** 贝塞尔校正值 (通常 0 或 1) */
    correction: number;
    /** 输入/输出 rank */
    rank: number;
}

/**
 * Welford Uniform Buffer 参数 (Global Reduction)
 */
export interface WelfordGlobalParams {
    /** 输入元素总数 */
    numel: number;
    /** 贝塞尔校正值 */
    correction: number;
    /** 输入 rank (用于 strided access) */
    rank: number;
}
