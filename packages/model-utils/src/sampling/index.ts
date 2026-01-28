/**
 * Sampling 模块 - 文本生成采样策略
 *
 * 实现 LLM 推理常用的采样策略：
 * - Temperature: 控制生成多样性
 * - Top-K: 限制候选词数量
 * - Top-P (Nucleus): 根据累积概率动态截断
 *
 * 对标 HuggingFace Transformers 的采样实现
 *
 * @module @kandle/model-utils/sampling
 */

import { Tensor, div, softmax } from "@kandle/core";

// ============================================================================
// Types
// ============================================================================

/**
 * 采样器配置
 */
export interface SamplerConfig {
    /**
     * Temperature - 控制生成多样性
     * - 值 < 1.0: 更确定性，倾向高概率token
     * - 值 = 1.0: 原始概率分布
     * - 值 > 1.0: 更随机，增加低概率token的机会
     * @default 1.0
     */
    temperature?: number;

    /**
     * Top-K - 只保留概率最高的K个token
     * - 值 = 0 或 undefined: 不限制
     * - 值 > 0: 只从前K个token中采样
     * @default 0
     */
    topK?: number;

    /**
     * Top-P (Nucleus Sampling) - 动态截断
     * - 保留累积概率达到P的最小token集合
     * - 值 = 1.0: 不截断
     * - 典型值: 0.9, 0.95
     * @default 1.0
     */
    topP?: number;

    /**
     * 是否使用贪婪采样 (argmax)
     * - 如果为 true，忽略其他采样参数
     * @default false
     */
    doSample?: boolean;
}

/**
 * 采样结果
 */
export interface SampleResult {
    /** 选中的 token ID */
    tokenId: number;

    /** 选中 token 的 log 概率 */
    logProb: number;

    /** Top-K 候选 token IDs (按概率降序) */
    topKTokenIds: number[];

    /** Top-K 候选 token 概率 (按概率降序) */
    topKProbs: Float32Array;
}

/**
 * 默认采样配置
 */
export const DEFAULT_SAMPLER_CONFIG: Required<SamplerConfig> = {
    temperature: 1.0,
    topK: 0,
    topP: 1.0,
    doSample: true,
};

// ============================================================================
// Implementation
// ============================================================================

/**
 * 从 logits 采样下一个 token
 *
 * @param logits - 模型输出的 logits，形状 [vocab_size] 或 [1, vocab_size]
 * @param config - 采样配置
 * @param displayTopK - 返回结果中包含的 top-k 候选数量 (默认 10)
 * @returns 采样结果，包含选中token和概率信息
 *
 * @example
 * ```typescript
 * const result = await sample(logits, { temperature: 0.7, topK: 50 });
 * console.log(`Selected token: ${result.tokenId}, prob: ${result.topKProbs[0]}`);
 * ```
 */
export async function sample(
    logits: Tensor,
    config: SamplerConfig = {},
    displayTopK: number = 10
): Promise<SampleResult> {
    const {
        temperature = DEFAULT_SAMPLER_CONFIG.temperature,
        topK = DEFAULT_SAMPLER_CONFIG.topK,
        topP = DEFAULT_SAMPLER_CONFIG.topP,
        doSample = DEFAULT_SAMPLER_CONFIG.doSample,
    } = config;

    // 获取 logits 数据到 CPU
    let logitsData = (await logits.dataAsync()) as Float32Array;

    // 如果是 2D [1, vocab_size]，提取第一个 batch
    if (logits.shape.length === 2) {
        logitsData = logitsData.slice(0, logits.shape[1]);
    }

    const vocabSize = logitsData.length;

    // 创建索引数组用于排序
    const indices = new Int32Array(vocabSize);
    for (let i = 0; i < vocabSize; i++) {
        indices[i] = i;
    }

    // 按 logits 值降序排序索引
    const sortedIndices = Array.from(indices).sort((a, b) => logitsData[b] - logitsData[a]);

    // ========================================
    // 贪婪模式 (argmax)
    // ========================================
    if (!doSample) {
        const tokenId = sortedIndices[0];

        // 计算 softmax 概率用于显示
        const probs = softmaxCPU(logitsData, temperature);

        // 收集 displayTopK 个候选
        const topKTokenIds: number[] = [];
        const topKProbsArr: number[] = [];
        for (let i = 0; i < Math.min(displayTopK, vocabSize); i++) {
            const idx = sortedIndices[i];
            topKTokenIds.push(idx);
            topKProbsArr.push(probs[idx]);
        }

        return {
            tokenId,
            logProb: Math.log(probs[tokenId] + 1e-10),
            topKTokenIds,
            topKProbs: new Float32Array(topKProbsArr),
        };
    }

    // ========================================
    // 概率采样模式
    // ========================================

    // Step 1: 应用 temperature
    let scaledLogits = logitsData;
    if (temperature !== 1.0) {
        scaledLogits = new Float32Array(vocabSize);
        const invTemp = 1.0 / temperature;
        for (let i = 0; i < vocabSize; i++) {
            scaledLogits[i] = logitsData[i] * invTemp;
        }
    }

    // Step 2: 计算 softmax 概率
    let probs = softmaxCPU(scaledLogits, 1.0); // temperature 已应用

    // Step 3: 应用 Top-K 过滤
    let effectiveK = vocabSize;
    if (topK > 0 && topK < vocabSize) {
        effectiveK = topK;
        // 将不在 top-K 的 token 概率设为 0
        const mask = new Float32Array(vocabSize);
        for (let i = 0; i < topK; i++) {
            mask[sortedIndices[i]] = 1.0;
        }
        for (let i = 0; i < vocabSize; i++) {
            probs[i] *= mask[i];
        }
    }

    // Step 4: 应用 Top-P (Nucleus) 过滤
    if (topP < 1.0) {
        let cumulativeProb = 0.0;
        let cutoffIdx = effectiveK;

        for (let i = 0; i < effectiveK; i++) {
            const idx = sortedIndices[i];
            cumulativeProb += probs[idx];
            if (cumulativeProb >= topP) {
                cutoffIdx = i + 1;
                break;
            }
        }

        // 将超过 cutoff 的 token 概率设为 0
        for (let i = cutoffIdx; i < vocabSize; i++) {
            const idx = sortedIndices[i];
            probs[idx] = 0.0;
        }
    }

    // Step 5: 重新归一化概率
    const probSum = probs.reduce((a, b) => a + b, 0);
    if (probSum > 0) {
        for (let i = 0; i < vocabSize; i++) {
            probs[i] /= probSum;
        }
    }

    // Step 6: 采样
    const tokenId = sampleFromProbs(probs);

    // 收集 displayTopK 个候选
    const topKTokenIds: number[] = [];
    const topKProbsArr: number[] = [];
    for (let i = 0; i < Math.min(displayTopK, vocabSize); i++) {
        const idx = sortedIndices[i];
        topKTokenIds.push(idx);
        topKProbsArr.push(probs[idx]);
    }

    return {
        tokenId,
        logProb: Math.log(probs[tokenId] + 1e-10),
        topKTokenIds,
        topKProbs: new Float32Array(topKProbsArr),
    };
}

/**
 * 贪婪采样 - 返回 argmax
 *
 * @param logits - 模型输出的 logits
 * @param displayTopK - 返回的 top-k 候选数量
 */
export async function sampleGreedy(
    logits: Tensor,
    displayTopK: number = 10
): Promise<SampleResult> {
    return sample(logits, { doSample: false }, displayTopK);
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * CPU 上的 softmax 实现
 */
function softmaxCPU(logits: Float32Array, temperature: number = 1.0): Float32Array {
    const n = logits.length;
    const result = new Float32Array(n);

    // 数值稳定性：减去最大值
    let maxLogit = -Infinity;
    for (let i = 0; i < n; i++) {
        if (logits[i] > maxLogit) maxLogit = logits[i];
    }

    // 计算 exp 并累加
    let sumExp = 0;
    const invTemp = 1.0 / temperature;
    for (let i = 0; i < n; i++) {
        const scaled = (logits[i] - maxLogit) * invTemp;
        result[i] = Math.exp(scaled);
        sumExp += result[i];
    }

    // 归一化
    for (let i = 0; i < n; i++) {
        result[i] /= sumExp;
    }

    return result;
}

/**
 * 从概率分布中采样一个索引
 * 使用简单的线性搜索 (对于 vocab_size ~= 150k 仍然很快)
 */
function sampleFromProbs(probs: Float32Array): number {
    const r = Math.random();
    let cumulative = 0;

    for (let i = 0; i < probs.length; i++) {
        cumulative += probs[i];
        if (r < cumulative) {
            return i;
        }
    }

    // 边界情况：返回最后一个非零概率的 token
    for (let i = probs.length - 1; i >= 0; i--) {
        if (probs[i] > 0) return i;
    }

    return 0;
}
