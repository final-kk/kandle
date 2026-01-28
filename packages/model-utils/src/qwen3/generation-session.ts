/**
 * GenerationSession - 有状态的文本生成会话
 *
 * 支持：
 * - 单步前进 (step)
 * - 后退到上一步 (undo)
 * - 干预选择 (override token)
 *
 * 与 generateStream 的区别：
 * - generateStream 是 AsyncGenerator，只能前进
 * - GenerationSession 是有状态类，支持双向操作
 *
 * @module @kandle/model-utils/qwen3/generation-session
 */

import { Tensor, select, softmax } from "@kandle/core";
import type { Qwen3ForCausalLM } from "./for_causal_lm";
import { StaticKVCache } from "../kv-cache";
import { sample, type SamplerConfig, type SampleResult } from "../sampling";

// ============================================================================
// Types
// ============================================================================

/**
 * Logit Lens 配置
 */
export interface LogitLensConfig {
    /** 是否启用 */
    enabled: boolean;
    /** 需要收集的层索引 */
    layerIndices: number[];
    /** 每层返回的 top-k 数量 */
    topK: number;
}

/**
 * 单层的 Logit Lens 预测结果
 */
export interface LayerPrediction {
    /** 层索引 */
    layerIndex: number;
    /** Top-K token IDs */
    topKTokenIds: number[];
    /** Top-K 概率 */
    topKProbs: Float32Array;
}

/**
 * GenerationSession 配置
 */
export interface GenerationSessionConfig extends SamplerConfig {
    /**
     * 最大生成 token 数量
     * @default 256
     */
    maxNewTokens?: number;

    /**
     * 结束 token ID 列表
     */
    eosTokenIds?: number[];

    /**
     * 返回的 Top-K 候选数量 (用于 UI 显示)
     * @default 10
     */
    displayTopK?: number;

    /**
     * Logit Lens 配置
     */
    logitLens?: LogitLensConfig;
}

/**
 * 生成步骤结果
 */
export interface SessionGenerationStep {
    /** 选中的 token ID */
    tokenId: number;

    /** 选中 token 的 log 概率 */
    logProb: number;

    /** Top-K 候选 token IDs (按概率降序) */
    topKTokenIds: number[];

    /** Top-K 候选 token 概率 (按概率降序) */
    topKProbs: Float32Array;

    /** 是否为结束 token */
    isEos: boolean;

    /** 当前 KV Cache 位置 */
    cachePosition: number;

    /** 已生成的 token 总数 */
    generatedCount: number;

    /** 是否可以后退 */
    canUndo: boolean;

    /** Logit Lens 各层预测结果 (可选) */
    logitLens?: LayerPrediction[];
}

/**
 * 默认配置
 */
const DEFAULT_SESSION_CONFIG: Required<GenerationSessionConfig> = {
    maxNewTokens: 256,
    eosTokenIds: [],
    displayTopK: 10,
    temperature: 1.0,
    topK: 0,
    topP: 1.0,
    doSample: true,
    logitLens: {
        enabled: false,
        layerIndices: [],
        topK: 3,
    },
};

// ============================================================================
// GenerationSession Class
// ============================================================================

/**
 * GenerationSession - 有状态的文本生成会话
 *
 * 生命周期：
 * 1. 创建 session: new GenerationSession(model, inputIds, config)
 * 2. 初始化 (prefill): await session.init()
 * 3. 单步生成: await session.step(overrideTokenId?)
 * 4. 后退: await session.undo()
 * 5. 释放资源: session.dispose()
 *
 * @example
 * ```typescript
 * const session = new GenerationSession(model, inputIds, {
 *     temperature: 0.7,
 *     topK: 50,
 * });
 *
 * // 初始化并获取第一步预测
 * const firstStep = await session.init();
 * console.log('First prediction:', firstStep.topKTokenIds);
 *
 * // 用户选择第二个候选
 * const step2 = await session.step(firstStep.topKTokenIds[1]);
 *
 * // 用户后悔了，回退
 * const undoStep = await session.undo();
 *
 * // 这次选择第一个候选
 * const step2Alt = await session.step();
 *
 * session.dispose();
 * ```
 */
export class GenerationSession {
    private model: Qwen3ForCausalLM;
    private kvCache: StaticKVCache;
    private config: Required<GenerationSessionConfig>;

    /** Prompt 的 token 长度 */
    private promptLength: number = 0;

    /** 当前 KV Cache 写入位置 */
    private cachePosition: number = 0;

    /** 已生成的 token ID 列表 */
    private generatedTokenIds: number[] = [];

    /** 当前位置的 logits (用于采样) */
    private currentLogits: Tensor | null = null;

    /** 当前位置各层的 logits (用于 Logit Lens) */
    private currentLayerLogits: Map<number, Tensor> | null = null;

    /** 当前步骤的采样结果 (缓存) */
    private currentSampleResult: SampleResult | null = null;

    /** 是否已初始化 */
    private initialized: boolean = false;

    /** 是否已结束 */
    private finished: boolean = false;

    constructor(
        model: Qwen3ForCausalLM,
        private inputIds: Tensor,
        config: GenerationSessionConfig = {}
    ) {
        this.model = model;
        this.config = { ...DEFAULT_SESSION_CONFIG, ...config };

        // 创建 KV Cache
        const seqLen = inputIds.shape[1];
        const maxSeqLen = seqLen + this.config.maxNewTokens + 16;
        this.kvCache = model.createKVCache(maxSeqLen);
    }

    /**
     * 初始化会话 (Prefill)
     *
     * 处理 prompt 并返回第一个预测步骤
     */
    async init(): Promise<SessionGenerationStep> {
        if (this.initialized) {
            throw new Error("Session already initialized");
        }

        const logitLensEnabled =
            this.config.logitLens.enabled && this.config.logitLens.layerIndices.length > 0;

        let logits: Tensor;
        let layerLogits: Map<number, Tensor> | undefined;
        let newCachePosition: number;

        if (logitLensEnabled) {
            // 使用带 Logit Lens 的前向传播
            const result = await this.model.forwardWithLogitLens(
                this.inputIds,
                this.kvCache,
                0,
                this.config.logitLens.layerIndices
            );
            logits = result.logits;
            layerLogits = result.layerLogits;
            newCachePosition = result.newCachePosition;
        } else {
            // 普通前向传播
            const result = await this.model.forwardWithKVCache(this.inputIds, this.kvCache, 0);
            logits = result.logits;
            newCachePosition = result.newCachePosition;
        }

        this.promptLength = this.inputIds.shape[1];
        this.cachePosition = newCachePosition;
        this.initialized = true;

        // 获取最后一个位置的 logits
        const seqLen = logits.shape[1];
        this.currentLogits = select(logits, 1, seqLen - 1);
        logits.dispose();

        // 处理各层 logits（获取最后一个位置）
        if (layerLogits) {
            this.disposeLayerLogits();
            this.currentLayerLogits = new Map();
            for (const [layerIdx, layerLogit] of layerLogits) {
                // select 返回视图，需要 clone 以获得独立副本
                // 否则 dispose layerLogit 后视图数据也会失效
                const lastPosView = select(layerLogit, 1, seqLen - 1);
                const lastPos = lastPosView.clone();
                lastPosView.dispose();
                this.currentLayerLogits.set(layerIdx, lastPos);
                layerLogit.dispose();
            }
        }

        // 采样
        return this.sampleAndBuildStep();
    }

    /**
     * 单步生成
     *
     * @param overrideTokenId 可选的干预 token ID，如果不提供则使用采样结果
     * @returns 下一步的预测结果，如果已结束则返回 null
     */
    async step(overrideTokenId?: number): Promise<SessionGenerationStep | null> {
        if (!this.initialized) {
            throw new Error("Session not initialized. Call init() first.");
        }

        if (this.finished) {
            return null;
        }

        if (!this.currentSampleResult) {
            throw new Error("No current sample result. This should not happen.");
        }

        // 确定最终使用的 token
        const actualTokenId = overrideTokenId ?? this.currentSampleResult.tokenId;

        // 记录生成的 token
        this.generatedTokenIds.push(actualTokenId);

        // 检查 EOS
        if (this.config.eosTokenIds.includes(actualTokenId)) {
            this.finished = true;
            this.disposeCurrentLogits();

            return {
                tokenId: actualTokenId,
                logProb: this.currentSampleResult.logProb,
                topKTokenIds: this.currentSampleResult.topKTokenIds,
                topKProbs: this.currentSampleResult.topKProbs,
                isEos: true,
                cachePosition: this.cachePosition,
                generatedCount: this.generatedTokenIds.length,
                canUndo: this.generatedTokenIds.length > 0,
            };
        }

        // 检查最大长度
        if (this.generatedTokenIds.length >= this.config.maxNewTokens) {
            this.finished = true;
            this.disposeCurrentLogits();
            return null;
        }

        // 准备下一步输入
        const nextInput = new Tensor(new Int32Array([actualTokenId]), {
            dtype: "int32",
            shape: [1, 1],
        });

        const logitLensEnabled =
            this.config.logitLens.enabled && this.config.logitLens.layerIndices.length > 0;

        let logits: Tensor;
        let layerLogits: Map<number, Tensor> | undefined;
        let newCachePosition: number;

        if (logitLensEnabled) {
            // 使用带 Logit Lens 的前向传播
            const result = await this.model.forwardWithLogitLens(
                nextInput,
                this.kvCache,
                this.cachePosition,
                this.config.logitLens.layerIndices
            );
            logits = result.logits;
            layerLogits = result.layerLogits;
            newCachePosition = result.newCachePosition;
        } else {
            // 普通前向传播
            const result = await this.model.forwardWithKVCache(
                nextInput,
                this.kvCache,
                this.cachePosition
            );
            logits = result.logits;
            newCachePosition = result.newCachePosition;
        }
        nextInput.dispose();

        this.cachePosition = newCachePosition;

        // 更新 logits
        this.disposeCurrentLogits();
        this.currentLogits = select(logits, 1, 0);
        logits.dispose();

        // 处理各层 logits（单 token 输入，直接使用）
        if (layerLogits) {
            // 先释放旧的 layer logits
            this.disposeLayerLogits();
            this.currentLayerLogits = new Map();
            for (const [layerIdx, layerLogit] of layerLogits) {
                // 对于单 token 输入，seq_len = 1，select dim=1, idx=0
                // select 返回视图，需要 clone 以获得独立副本
                const lastPosView = select(layerLogit, 1, 0);
                const lastPos = lastPosView.clone();
                lastPosView.dispose();
                this.currentLayerLogits.set(layerIdx, lastPos);
                layerLogit.dispose();
            }
        }

        // 采样
        return this.sampleAndBuildStep();
    }

    /**
     * 后退一步
     *
     * 回滚到上一个状态，重新获取该位置的预测
     *
     * @returns 回退后的预测结果，如果无法再后退则返回 null
     */
    async undo(): Promise<SessionGenerationStep | null> {
        if (!this.initialized) {
            throw new Error("Session not initialized. Call init() first.");
        }

        if (this.generatedTokenIds.length === 0) {
            // 无法再后退
            return null;
        }

        // 弹出最后一个 token
        this.generatedTokenIds.pop();

        // 如果已结束，重置结束状态
        this.finished = false;

        // 计算回滚目标位置
        const targetPosition = this.promptLength + this.generatedTokenIds.length;

        // 回滚 KV Cache
        this.kvCache.rollback(targetPosition);
        this.cachePosition = targetPosition;

        // 确定用于重新计算 logits 的 token
        let tokenIdForRegen: number;
        if (this.generatedTokenIds.length > 0) {
            // 使用生成列表中最后一个 token
            tokenIdForRegen = this.generatedTokenIds[this.generatedTokenIds.length - 1];
        } else {
            // 回退到 prompt 结束位置，需要使用 prompt 的最后一个 token
            // 但是我们已经 prefill 过了，需要重新获取 prompt 最后一个 token 的 logits
            // 实际上，prefill 时已经计算了 prompt 最后位置的 logits
            // 这里需要重新运行 prefill 的最后一步...

            // 方案：重新读取 prompt 的最后一个 token 并 forward
            // 但这会导致重复计算，更好的方案是缓存 prefill 后的 logits

            // 简化处理：如果没有生成任何 token，重新 prefill
            // 这种情况应该很少发生
            return this.reinitFromPrompt();
        }

        // 重新计算 logits
        const input = new Tensor(new Int32Array([tokenIdForRegen]), {
            dtype: "int32",
            shape: [1, 1],
        });

        const logitLensEnabled =
            this.config.logitLens.enabled && this.config.logitLens.layerIndices.length > 0;

        let logits: Tensor;
        let layerLogits: Map<number, Tensor> | undefined;
        let newCachePosition: number;

        // 注意：这里需要使用 targetPosition - 1 作为起始位置
        // 因为我们要重新写入这个 token 的 KV
        if (logitLensEnabled) {
            const result = await this.model.forwardWithLogitLens(
                input,
                this.kvCache,
                targetPosition - 1,
                this.config.logitLens.layerIndices
            );
            logits = result.logits;
            layerLogits = result.layerLogits;
            newCachePosition = result.newCachePosition;
        } else {
            const result = await this.model.forwardWithKVCache(
                input,
                this.kvCache,
                targetPosition - 1
            );
            logits = result.logits;
            newCachePosition = result.newCachePosition;
        }
        input.dispose();

        this.cachePosition = newCachePosition;

        // 更新 logits
        this.disposeCurrentLogits();
        this.currentLogits = select(logits, 1, 0);
        logits.dispose();

        // 处理各层 logits
        if (layerLogits) {
            this.disposeLayerLogits();
            this.currentLayerLogits = new Map();
            for (const [layerIdx, layerLogit] of layerLogits) {
                const lastPosView = select(layerLogit, 1, 0);
                const lastPos = lastPosView.clone();
                lastPosView.dispose();
                this.currentLayerLogits.set(layerIdx, lastPos);
                layerLogit.dispose();
            }
        }

        // 采样
        return this.sampleAndBuildStep();
    }

    /**
     * 获取当前状态
     */
    getState(): {
        generatedCount: number;
        canUndo: boolean;
        isFinished: boolean;
        cachePosition: number;
    } {
        return {
            generatedCount: this.generatedTokenIds.length,
            canUndo: this.generatedTokenIds.length > 0,
            isFinished: this.finished,
            cachePosition: this.cachePosition,
        };
    }

    /**
     * 获取已生成的 token IDs
     */
    getGeneratedTokenIds(): number[] {
        return [...this.generatedTokenIds];
    }

    /**
     * 释放资源
     */
    dispose(): void {
        this.disposeCurrentLogits();
        this.kvCache.dispose();
        this.initialized = false;
        this.finished = true;
    }

    // ========================================
    // Private Methods
    // ========================================

    /**
     * 采样并构建步骤结果
     */
    private async sampleAndBuildStep(): Promise<SessionGenerationStep> {
        if (!this.currentLogits) {
            throw new Error("No current logits available");
        }

        // 采样
        const samplerConfig: SamplerConfig = {
            temperature: this.config.temperature,
            topK: this.config.topK,
            topP: this.config.topP,
            doSample: this.config.doSample,
        };

        this.currentSampleResult = await sample(
            this.currentLogits,
            samplerConfig,
            this.config.displayTopK
        );

        const isEos = this.config.eosTokenIds.includes(this.currentSampleResult.tokenId);

        // 处理 Logit Lens 结果
        let logitLensResults: LayerPrediction[] | undefined;
        if (this.currentLayerLogits && this.currentLayerLogits.size > 0) {
            logitLensResults = await this.computeLogitLensPredictions();
        }

        return {
            tokenId: this.currentSampleResult.tokenId,
            logProb: this.currentSampleResult.logProb,
            topKTokenIds: this.currentSampleResult.topKTokenIds,
            topKProbs: this.currentSampleResult.topKProbs,
            isEos,
            cachePosition: this.cachePosition,
            generatedCount: this.generatedTokenIds.length,
            canUndo: this.generatedTokenIds.length > 0,
            logitLens: logitLensResults,
        };
    }

    /**
     * 计算 Logit Lens 各层的 top-k 预测
     */
    private async computeLogitLensPredictions(): Promise<LayerPrediction[]> {
        if (!this.currentLayerLogits) {
            return [];
        }

        const predictions: LayerPrediction[] = [];
        const topK = this.config.logitLens.topK;

        // 按层索引排序
        const sortedLayers = Array.from(this.currentLayerLogits.entries()).sort(
            (a, b) => a[0] - b[0]
        );

        for (const [layerIndex, layerLogit] of sortedLayers) {
            // 应用 softmax 获取概率
            const probs = softmax(layerLogit, -1);

            // 获取 top-k
            const probsData = await probs.dataAsync();
            probs.dispose();

            // 找到 top-k 索引和概率 (确保是 Float32Array)
            const probsFloat32 =
                probsData instanceof Float32Array
                    ? probsData
                    : new Float32Array(probsData as ArrayLike<number>);
            const indexed = Array.from(probsFloat32).map((prob, idx) => ({ prob, idx }));
            indexed.sort((a, b) => b.prob - a.prob);

            const topKItems = indexed.slice(0, topK);
            const topKTokenIds = topKItems.map((item) => item.idx);
            const topKProbs = new Float32Array(topKItems.map((item) => item.prob));

            predictions.push({
                layerIndex,
                topKTokenIds,
                topKProbs,
            });
        }

        return predictions;
    }

    /**
     * 从 prompt 重新初始化 (用于 undo 到初始状态)
     */
    private async reinitFromPrompt(): Promise<SessionGenerationStep> {
        // 重置 KV Cache
        this.kvCache.reset();

        const logitLensEnabled =
            this.config.logitLens.enabled && this.config.logitLens.layerIndices.length > 0;

        let logits: Tensor;
        let layerLogits: Map<number, Tensor> | undefined;
        let newCachePosition: number;

        // 重新 prefill
        if (logitLensEnabled) {
            const result = await this.model.forwardWithLogitLens(
                this.inputIds,
                this.kvCache,
                0,
                this.config.logitLens.layerIndices
            );
            logits = result.logits;
            layerLogits = result.layerLogits;
            newCachePosition = result.newCachePosition;
        } else {
            const result = await this.model.forwardWithKVCache(this.inputIds, this.kvCache, 0);
            logits = result.logits;
            newCachePosition = result.newCachePosition;
        }

        this.cachePosition = newCachePosition;

        // 获取最后一个位置的 logits
        const seqLen = logits.shape[1];
        this.disposeCurrentLogits();
        this.currentLogits = select(logits, 1, seqLen - 1);
        logits.dispose();

        // 处理各层 logits
        if (layerLogits) {
            this.disposeLayerLogits();
            this.currentLayerLogits = new Map();
            for (const [layerIdx, layerLogit] of layerLogits) {
                const lastPosView = select(layerLogit, 1, seqLen - 1);
                const lastPos = lastPosView.clone();
                lastPosView.dispose();
                this.currentLayerLogits.set(layerIdx, lastPos);
                layerLogit.dispose();
            }
        }

        // 采样
        return this.sampleAndBuildStep();
    }

    /**
     * 释放当前 logits
     */
    private disposeCurrentLogits(): void {
        if (this.currentLogits) {
            this.currentLogits.dispose();
            this.currentLogits = null;
        }
        this.currentSampleResult = null;
        // 同时释放 layer logits
        this.disposeLayerLogits();
    }

    /**
     * 释放各层 logits
     */
    private disposeLayerLogits(): void {
        if (this.currentLayerLogits) {
            for (const tensor of this.currentLayerLogits.values()) {
                tensor.dispose();
            }
            this.currentLayerLogits = null;
        }
    }
}
