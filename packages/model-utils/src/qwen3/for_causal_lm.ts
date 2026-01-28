/**
 * Qwen3ForCausalLM - Qwen3 因果语言模型
 *
 * 包含语言模型头（LM Head）的完整 Qwen3 模型，用于文本生成任务。
 *
 * @module @kandle/model-utils/qwen3/for_causal_lm
 */

import { Tensor, nn, matmul, transpose, select } from "@kandle/core";
import { Qwen3Model, type Qwen3ModelConfig } from "./model";
import { StaticKVCache } from "../kv-cache";
import { sample, type SamplerConfig, type SampleResult } from "../sampling";

// ============================================================================
// Types
// ============================================================================

/**
 * 生成步骤结果 - 每一步 generator 返回的信息
 */
export interface GenerationStep {
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
}

/**
 * Generator 配置
 */
export interface GeneratorConfig extends SamplerConfig {
    /**
     * 最大生成 token 数量
     * @default 256
     */
    maxNewTokens?: number;

    /**
     * 结束 token ID 列表
     * 遇到这些 token 时停止生成
     */
    eosTokenIds?: number[];

    /**
     * 返回的 Top-K 候选数量 (用于 UI 显示)
     * @default 10
     */
    displayTopK?: number;
}

/**
 * 默认 Generator 配置
 */
export const DEFAULT_GENERATOR_CONFIG: Required<GeneratorConfig> = {
    maxNewTokens: 256,
    eosTokenIds: [],
    displayTopK: 10,
    temperature: 1.0,
    topK: 0,
    topP: 1.0,
    doSample: true,
};

// ============================================================================
// Qwen3ForCausalLM Class
// ============================================================================

/**
 * Qwen3 因果语言模型
 *
 * 在 Qwen3Model 基础上添加语言模型头，支持：
 * - 标准前向传播（forward）
 * - 单 token 生成（generateNextToken）
 * - 带 KV Cache 的高效推理（generateWithKVCache）
 * - **流式生成器 (generateStream)** - 支持逐步生成和外部干预
 */
export class Qwen3ForCausalLM extends nn.Module {
    model: Qwen3Model;
    private lmHeadWeight: Tensor | null = null;
    private usesTiedEmbeddings: boolean;

    // 模型配置，用于创建 KV Cache
    readonly config: Qwen3ModelConfig;

    /**
     * 构造 Qwen3ForCausalLM
     *
     * @param config - Qwen3 模型配置
     * @param usesTiedEmbeddings - 是否使用绑定的词嵌入权重（LM Head 与 Embedding 共享权重）
     */
    constructor(config: Qwen3ModelConfig, usesTiedEmbeddings = true) {
        super();
        this.config = config;
        this.model = new Qwen3Model(config);
        this.usesTiedEmbeddings = usesTiedEmbeddings;
        this.addModule("model", this.model);
    }

    /**
     * 初始化语言模型头
     *
     * 必须在加载权重后调用，用于设置 LM Head 权重。
     * 如果使用绑定的词嵌入（usesTiedEmbeddings=true），则 LM Head 与 Embedding 共享权重。
     */
    initLMHead(): void {
        if (this.usesTiedEmbeddings) {
            this.lmHeadWeight = this.model.embed_tokens.weight;
        }
    }

    /**
     * 创建适配当前模型的 KV Cache
     */
    createKVCache(maxSeqLen: number = 2048): StaticKVCache {
        // Qwen3 可能有显式的 headDim 配置，不一定等于 hiddenSize / numAttentionHeads
        const headDim =
            this.config.headDim ??
            Math.floor(this.config.hiddenSize / this.config.numAttentionHeads);
        return new StaticKVCache({
            numLayers: this.config.numHiddenLayers,
            numKvHeads: this.config.numKeyValueHeads,
            headDim,
            maxSeqLen,
        });
    }

    /**
     * 前向传播
     *
     * @param inputIds - 输入 token IDs，形状 [batch_size, seq_len]
     * @returns logits，形状 [batch_size, seq_len, vocab_size]
     */
    async forward(inputIds: Tensor): Promise<Tensor> {
        const hiddenStates = (await this.model.call(inputIds)) as Tensor;

        if (this.lmHeadWeight === null) {
            throw new Error("LM head not initialized. Call initLMHead() after loading weights.");
        }

        const transposed = transpose(this.lmHeadWeight, 0, 1);
        const logits = matmul(hiddenStates, transposed);

        return logits;
    }

    /**
     * 使用 KV Cache 进行一步推理，返回 logits
     *
     * @param inputIds - 输入 token IDs，形状 [batch_size, seq_len]
     * @param kvCache - KV Cache 实例
     * @param cachePosition - 当前缓存位置
     * @returns logits 和新的缓存位置
     */
    async forwardWithKVCache(
        inputIds: Tensor,
        kvCache: StaticKVCache,
        cachePosition: number
    ): Promise<{ logits: Tensor; newCachePosition: number }> {
        const hiddenStates = (await this.model.call(inputIds, {
            kvCache,
            cachePosition,
        })) as Tensor;

        if (this.lmHeadWeight === null) {
            throw new Error("LM head not initialized. Call initLMHead() after loading weights.");
        }

        const transposed = transpose(this.lmHeadWeight, 0, 1);
        const logits = matmul(hiddenStates, transposed);
        hiddenStates.dispose();

        const seqLen = inputIds.shape[1];

        return {
            logits,
            newCachePosition: cachePosition + seqLen,
        };
    }

    /**
     * 使用 KV Cache 进行一步推理，同时收集各层的 Logit Lens 结果
     *
     * @param inputIds - 输入 token IDs，形状 [batch_size, seq_len]
     * @param kvCache - KV Cache 实例
     * @param cachePosition - 当前缓存位置
     * @param collectLayerIndices - 需要收集 Logit Lens 的层索引
     * @returns logits、各层 logits 和新的缓存位置
     */
    async forwardWithLogitLens(
        inputIds: Tensor,
        kvCache: StaticKVCache,
        cachePosition: number,
        collectLayerIndices: number[]
    ): Promise<{
        logits: Tensor;
        layerLogits: Map<number, Tensor>;
        newCachePosition: number;
    }> {
        // 使用带层输出收集的前向传播
        const { hiddenStates, layerHiddenStates } = await this.model.forwardWithLayerOutputs(
            inputIds,
            {
                kvCache,
                cachePosition,
                collectLayerIndices,
            }
        );

        if (this.lmHeadWeight === null) {
            throw new Error("LM head not initialized. Call initLMHead() after loading weights.");
        }

        const transposed = transpose(this.lmHeadWeight, 0, 1);

        // 计算最终 logits
        const logits = matmul(hiddenStates, transposed);
        hiddenStates.dispose();

        // 计算各层的 Logit Lens logits
        const layerLogits = new Map<number, Tensor>();
        if (layerHiddenStates) {
            for (const [layerIdx, layerHidden] of layerHiddenStates) {
                // 对各层 hidden states 应用 final norm 后再投影
                // 这是 Logit Lens 的标准做法，确保分布一致
                const normed = (await this.model.norm.call(layerHidden)) as Tensor;
                const layerLogit = matmul(normed, transposed);
                normed.dispose();
                // layerHidden 是 clone 的副本，需要释放
                layerHidden.dispose();
                layerLogits.set(layerIdx, layerLogit);
            }
        }

        const seqLen = inputIds.shape[1];

        return {
            logits,
            layerLogits,
            newCachePosition: cachePosition + seqLen,
        };
    }

    /**
     * 流式生成器 - 逐步生成 token
     *
     * 这是核心的可解释性 API，支持：
     * 1. 逐步生成，每步返回完整的采样信息（概率分布、候选词等）
     * 2. 外部干预：通过 generator.next(overrideTokenId) 指定下一个 token
     * 3. 暂停/继续：外部控制调用 next() 的时机
     *
     * @param inputIds - 初始输入 token IDs，形状 [1, seq_len]
     * @param config - 生成配置
     * @yields GenerationStep - 每步的生成信息
     *
     * @example
     * ```typescript
     * const generator = model.generateStream(inputIds, { temperature: 0.7 });
     *
     * // 自动生成
     * for await (const step of generator) {
     *     console.log(`Token: ${step.tokenId}, Prob: ${step.topKProbs[0]}`);
     *     if (step.isEos) break;
     * }
     *
     * // 手动控制
     * let result = await generator.next();
     * while (!result.done) {
     *     const step = result.value;
     *     // 用户选择干预：使用第二个候选词
     *     const overrideToken = step.topKTokenIds[1];
     *     result = await generator.next(overrideToken);
     * }
     * ```
     */
    async *generateStream(
        inputIds: Tensor,
        config: GeneratorConfig = {}
    ): AsyncGenerator<GenerationStep, void, number | undefined> {
        const {
            maxNewTokens = DEFAULT_GENERATOR_CONFIG.maxNewTokens,
            eosTokenIds = DEFAULT_GENERATOR_CONFIG.eosTokenIds,
            displayTopK = DEFAULT_GENERATOR_CONFIG.displayTopK,
            temperature = DEFAULT_GENERATOR_CONFIG.temperature,
            topK = DEFAULT_GENERATOR_CONFIG.topK,
            topP = DEFAULT_GENERATOR_CONFIG.topP,
            doSample = DEFAULT_GENERATOR_CONFIG.doSample,
        } = config;

        // 创建 KV Cache
        const maxSeqLen = inputIds.shape[1] + maxNewTokens + 16;
        const kvCache = this.createKVCache(maxSeqLen);

        // Prefill: 处理初始输入
        let { logits, newCachePosition } = await this.forwardWithKVCache(inputIds, kvCache, 0);
        let cachePosition = newCachePosition;

        // 获取最后一个位置的 logits
        const seqLen = logits.shape[1];
        let lastLogits = select(logits, 1, seqLen - 1);
        logits.dispose();

        // 采样配置
        const samplerConfig: SamplerConfig = {
            temperature,
            topK,
            topP,
            doSample,
        };

        // 用于追踪生成的 token
        const generatedTokens: number[] = [];

        // 生成循环
        for (let i = 0; i < maxNewTokens; i++) {
            // 采样下一个 token
            const sampleResult = await sample(lastLogits, samplerConfig, displayTopK);
            lastLogits.dispose();

            // 检查是否有外部干预
            // yield 返回当前步骤信息，并等待外部调用 next()
            // 如果 next() 传入了 tokenId，则使用该 token
            const overrideTokenId: number | undefined = yield {
                tokenId: sampleResult.tokenId,
                logProb: sampleResult.logProb,
                topKTokenIds: sampleResult.topKTokenIds,
                topKProbs: sampleResult.topKProbs,
                isEos: eosTokenIds.includes(sampleResult.tokenId),
                cachePosition,
                generatedCount: i + 1,
            };

            // 确定最终使用的 token
            const actualTokenId =
                overrideTokenId !== undefined ? overrideTokenId : sampleResult.tokenId;
            generatedTokens.push(actualTokenId);

            // 检查 EOS
            if (eosTokenIds.includes(actualTokenId)) {
                break;
            }

            // 准备下一步输入
            const nextInput = new Tensor(new Int32Array([actualTokenId]), {
                dtype: "int32",
                shape: [1, 1],
            });

            // 前向传播
            const result = await this.forwardWithKVCache(nextInput, kvCache, cachePosition);
            nextInput.dispose();

            cachePosition = result.newCachePosition;

            // 获取 logits（单 token 输入，不需要 select）
            lastLogits = select(result.logits, 1, 0);
            result.logits.dispose();
        }

        // 清理 KV Cache
        kvCache.dispose();
    }

    /**
     * 生成下一个 token
     *
     * 使用简单的 argmax 采样策略。
     *
     * @param inputIds - 输入 token IDs，形状 [batch_size, seq_len]
     * @returns 下一个 token 的 ID
     */
    async generateNextToken(inputIds: Tensor): Promise<number> {
        const logits = (await this.call(inputIds)) as Tensor;

        // 获取最后一个位置的 logits
        const seqLen = logits.shape[1];
        const lastLogits = select(logits, 1, seqLen - 1);

        // 简单的 argmax 采样
        const argmaxResult = lastLogits.argmax(1);
        const gpuData = await argmaxResult.dataAsync();
        const gpuIdx = Number(gpuData[0]);

        return gpuIdx;
    }

    /**
     * 使用 KV Cache 生成下一个 token
     *
     * 支持高效的增量推理，避免重复计算历史 token 的注意力。
     *
     * @param inputIds - 输入 token IDs，形状 [batch_size, seq_len]
     * @param kvCache - KV Cache 实例
     * @param cachePosition - 当前缓存位置
     * @returns 下一个 token ID 和新的缓存位置
     */
    async generateWithKVCache(
        inputIds: Tensor,
        kvCache: StaticKVCache,
        cachePosition: number
    ): Promise<{ nextTokenId: number; newCachePosition: number }> {
        const hiddenStates = (await this.model.call(inputIds, {
            kvCache,
            cachePosition,
        })) as Tensor;

        if (this.lmHeadWeight === null) {
            throw new Error("LM head not initialized. Call initLMHead() after loading weights.");
        }

        const transposed = transpose(this.lmHeadWeight, 0, 1);
        const logits = matmul(hiddenStates, transposed);
        hiddenStates.dispose();

        const seqLen = logits.shape[1];
        const lastLogits = select(logits, 1, seqLen - 1);
        logits.dispose();

        const argmaxResult = lastLogits.argmax(1);
        lastLogits.dispose();

        const data = await argmaxResult.dataAsync();
        const nextTokenId = Number(data[0]);
        argmaxResult.dispose();

        return {
            nextTokenId,
            newCachePosition: cachePosition + seqLen,
        };
    }
}
