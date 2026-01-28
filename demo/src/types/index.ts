/**
 * Demo 应用类型定义
 */

// ============================================================================
// Generator 相关类型
// ============================================================================

/**
 * 生成步骤 - 每一步生成返回的信息
 */
export interface GenerationStep {
    /** 选中的 token ID */
    tokenId: number;
    /** 选中 token 的文本 */
    tokenText: string;
    /** 选中 token 的概率 */
    probability: number;
    /** 选中 token 的 log 概率 */
    logProb: number;
    /** Top-K 候选 token 信息 */
    topK: TokenCandidate[];
    /** 是否为结束 token */
    isEos: boolean;
    /** 当前 KV Cache 位置 */
    cachePosition: number;
    /** 已生成的 token 总数 */
    generatedCount: number;
    /** Logit Lens 各层预测结果（可选） */
    logitLens?: LayerPrediction[];
}

/**
 * 单层的 Logit Lens 预测结果
 */
export interface LayerPrediction {
    /** 层索引 */
    layerIndex: number;
    /** Top-K 候选列表 */
    topK: TokenCandidate[];
}

/**
 * Token 候选项
 */
export interface TokenCandidate {
    /** Token ID */
    tokenId: number;
    /** Token 文本 */
    text: string;
    /** 概率 */
    probability: number;
}

/**
 * 生成的 Token 记录
 */
export interface GeneratedToken {
    /** Token ID */
    id: number;
    /** Token 文本 */
    text: string;
    /** 概率 */
    probability: number;
    /** 是否由用户干预选择 */
    isOverride: boolean;
}

// ============================================================================
// 采样配置
// ============================================================================

/**
 * 采样器配置
 */
export interface SamplerConfig {
    /** Temperature - 控制生成多样性 */
    temperature: number;
    /** Top-K - 只保留概率最高的K个token */
    topK: number;
    /** Top-P (Nucleus) - 动态截断 */
    topP: number;
    /** 是否使用采样 (false = greedy) */
    doSample: boolean;
}

/**
 * 默认采样配置
 */
export const DEFAULT_SAMPLER_CONFIG: SamplerConfig = {
    temperature: 0.7,
    topK: 50,
    topP: 0.95,
    doSample: true,
};

// ============================================================================
// 应用状态
// ============================================================================

/**
 * 生成状态
 */
export type GeneratorStatus = "idle" | "generating" | "paused" | "complete" | "error";

/**
 * 模型加载状态
 */
export type ModelStatus = "unloaded" | "loading" | "loaded" | "error";

/**
 * 日志条目
 */
export interface LogEntry {
    id: number;
    timestamp: Date;
    message: string;
    type: "info" | "success" | "error" | "warning" | "kernel" | "debug";
}

// ============================================================================
// WebGPU 信息
// ============================================================================

/**
 * WebGPU 设备信息
 */
export interface WebGPUInfo {
    name: string;
    vendor: string;
    architecture: string;
    supportsF16: boolean;
    maxBufferSize: number;
    maxStorageBufferBindingSize: number;
}
