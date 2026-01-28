/**
 * Model Handler
 *
 * 在 Worker 中处理模型加载和推理逻辑
 * 使用 GenerationSession 支持前进/后退操作
 */

import { io, Tensor } from "@kandle/core";
import {
    Qwen3ForCausalLM,
    GenerationSession,
    type SessionGenerationStep,
} from "@kandle/model-utils";
import { Tokenizer } from "@huggingface/tokenizers";
import type {
    LoadProgress,
    GenerationStep,
    StartGenerationPayload,
    TokenCandidate,
    LoadModelFromBufferPayload,
    LayerPrediction,
} from "./message-types";

// ============================================================================
// Qwen3-0.6B 配置
// ============================================================================

const QWEN3_0_6B_CONFIG = {
    vocabSize: 151936,
    hiddenSize: 1024,
    intermediateSize: 3072,
    numHiddenLayers: 28,
    numAttentionHeads: 16,
    numKeyValueHeads: 8,
    headDim: 128,
    maxPositionEmbeddings: 40960,
    ropeTheta: 1000000,
    rmsNormEps: 1e-6,
    attentionBias: false,
    mlpBias: false,
    dtype: "float32" as const,
};

// ============================================================================
// 工具函数
// ============================================================================

/**
 * 带进度的 fetch
 */
async function fetchWithProgress(
    url: string,
    onProgress: (loaded: number, total: number) => void
): Promise<ArrayBuffer> {
    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
    }

    const contentLength = response.headers.get("content-length");
    const total = contentLength ? parseInt(contentLength, 10) : 0;

    if (!response.body) {
        const buffer = await response.arrayBuffer();
        onProgress(buffer.byteLength, buffer.byteLength);
        return buffer;
    }

    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        onProgress(loaded, total);
    }

    // 合并 chunks
    const buffer = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
        buffer.set(chunk, offset);
        offset += chunk.length;
    }

    return buffer.buffer;
}

// ============================================================================
// ModelHandler 类
// ============================================================================

export class ModelHandler {
    private model: Qwen3ForCausalLM | null = null;
    private tokenizer: Tokenizer | null = null;

    /** 当前生成会话 (支持前进/后退) */
    private session: GenerationSession | null = null;

    /** 模型是否已加载 */
    get modelLoaded(): boolean {
        return this.model !== null && this.tokenizer !== null;
    }

    /**
     * 从 URL 加载模型
     */
    async loadModel(
        tokenizerUrl: string,
        modelUrl: string,
        onProgress: (progress: LoadProgress) => void
    ): Promise<void> {
        console.log("[ModelHandler] Starting model load from URL...");

        // 1. 下载 tokenizer
        console.log("[ModelHandler] Downloading tokenizer...");
        onProgress({ stage: "tokenizer", loaded: 0, total: 1, fileName: "tokenizer.json" });

        const tokenizerResp = await fetch(tokenizerUrl);
        if (!tokenizerResp.ok) {
            throw new Error(`Failed to fetch tokenizer: ${tokenizerResp.status}`);
        }
        const tokenizerJson = await tokenizerResp.json();

        onProgress({ stage: "tokenizer", loaded: 1, total: 1, fileName: "tokenizer.json" });
        console.log("[ModelHandler] Tokenizer downloaded");

        // 2. 下载模型权重（带进度）
        console.log("[ModelHandler] Downloading model weights...");
        const modelBuffer = await fetchWithProgress(modelUrl, (loaded, total) => {
            onProgress({
                stage: "model",
                loaded,
                total,
                fileName: "model.safetensors",
                speed: 0, // TODO: 计算速度
            });
        });
        console.log("[ModelHandler] Model downloaded, size:", modelBuffer.byteLength);

        // 3. 初始化模型
        await this.initializeModel(tokenizerJson, modelBuffer, onProgress);
    }

    /**
     * 从 ArrayBuffer 加载模型（用于文件上传模式）
     */
    async loadModelFromBuffer(
        payload: LoadModelFromBufferPayload,
        onProgress: (progress: LoadProgress) => void
    ): Promise<void> {
        console.log("[ModelHandler] Starting model load from buffer...");

        // 1. 解析 tokenizer JSON
        onProgress({ stage: "tokenizer", loaded: 0, total: 1, fileName: "tokenizer.json" });

        const decoder = new TextDecoder();
        const tokenizerText = decoder.decode(payload.tokenizerBuffer);
        const tokenizerJson = JSON.parse(tokenizerText);

        onProgress({ stage: "tokenizer", loaded: 1, total: 1, fileName: "tokenizer.json" });
        console.log("[ModelHandler] Tokenizer parsed");

        // 2. 模型 buffer 已经准备好
        onProgress({
            stage: "model",
            loaded: payload.modelBuffer.byteLength,
            total: payload.modelBuffer.byteLength,
            fileName: "model.safetensors",
        });
        console.log("[ModelHandler] Model buffer ready, size:", payload.modelBuffer.byteLength);

        // 3. 初始化模型
        await this.initializeModel(tokenizerJson, payload.modelBuffer, onProgress);
    }

    /**
     * 初始化模型（共用逻辑）
     */
    private async initializeModel(
        tokenizerJson: Record<string, unknown>,
        modelBuffer: ArrayBuffer,
        onProgress: (progress: LoadProgress) => void
    ): Promise<void> {
        // 1. 创建 tokenizer
        this.tokenizer = new Tokenizer(tokenizerJson, {});
        console.log("[ModelHandler] Tokenizer initialized");

        // 2. 创建模型实例
        console.log("[ModelHandler] Creating model instance...");
        onProgress({ stage: "weights", loaded: 0, total: 100, fileName: "Creating model..." });

        this.model = new Qwen3ForCausalLM(QWEN3_0_6B_CONFIG, true);

        // 3. 加载权重
        console.log("[ModelHandler] Loading safetensor weights...");
        onProgress({ stage: "weights", loaded: 10, total: 100, fileName: "Parsing safetensor..." });

        const group = await io.loadSafetensor(modelBuffer);
        console.log(`[ModelHandler] Safetensor parsed: ${group.layers.size} layers`);

        onProgress({ stage: "weights", loaded: 30, total: 100, fileName: "Applying weights..." });

        const result = await this.model.loadFromSafetensor(group, {
            strict: false,
            keyMapper: (key: string) => key,
        });

        console.log(
            `[ModelHandler] Weights loaded: ${result.loadedKeys.length} loaded, ` +
                `${result.missingKeys.length} missing, ${result.unexpectedKeys.length} unexpected`
        );

        if (result.missingKeys.length > 0) {
            console.warn("[ModelHandler] Missing keys:", result.missingKeys.slice(0, 5));
        }

        // 4. 关闭 group 释放资源
        group.close();

        onProgress({
            stage: "weights",
            loaded: 80,
            total: 100,
            fileName: "Initializing LM head...",
        });

        // 5. 初始化 LM Head
        console.log("[ModelHandler] Initializing LM head...");
        this.model.initLMHead();

        onProgress({ stage: "weights", loaded: 100, total: 100, fileName: "Complete" });
        console.log("[ModelHandler] Model ready!");
    }

    /**
     * 开始生成（创建 GenerationSession 并返回第一步）
     *
     * 使用 GenerationSession 代替 AsyncGenerator，支持前进/后退操作
     */
    async startGeneration(config: StartGenerationPayload): Promise<GenerationStep> {
        if (!this.model || !this.tokenizer) {
            throw new Error("Model or tokenizer not loaded");
        }

        // 清理之前的 session
        if (this.session) {
            this.session.dispose();
            this.session = null;
        }

        console.log("[ModelHandler] Starting generation with config:", config);

        // Encode prompt
        const encoded = await this.tokenizer.encode(config.prompt);
        const inputIds = new Tensor(new Int32Array(encoded.ids), {
            dtype: "int32",
            shape: [1, encoded.ids.length],
        });

        console.log(`[ModelHandler] Prompt encoded: ${encoded.ids.length} tokens`);

        // 创建 GenerationSession
        // 注意: 使用类型断言处理跨包类型不匹配问题（包需要重新构建）
        this.session = new GenerationSession(
            this.model,
            inputIds as any,
            {
                temperature: config.temperature,
                topK: config.topK,
                topP: config.topP,
                doSample: config.doSample,
                eosTokenIds: config.eosTokenIds,
                displayTopK: config.displayTopK,
                maxNewTokens: config.maxNewTokens,
                logitLens: config.logitLens,
            } as any
        );

        // 初始化并获取第一步预测
        const firstStep = await this.session.init();

        const uiStep = await this.convertSessionStep(firstStep);
        console.log("[ModelHandler] First step ready:", uiStep.tokenText);

        return uiStep;
    }

    /**
     * 单步生成（前进）
     */
    async step(overrideTokenId?: number): Promise<GenerationStep | null> {
        if (!this.session) {
            console.warn("[ModelHandler] No active session for step");
            return null;
        }

        try {
            const result = await this.session.step(overrideTokenId);

            if (!result) {
                // 生成结束
                return null;
            }

            return await this.convertSessionStep(result);
        } catch (error) {
            console.error("[ModelHandler] Step error:", error);
            throw error;
        }
    }

    /**
     * 后退一步
     *
     * 回滚到上一个状态，重新获取该位置的预测
     */
    async undo(): Promise<GenerationStep | null> {
        if (!this.session) {
            console.warn("[ModelHandler] No active session for undo");
            return null;
        }

        try {
            const result = await this.session.undo();

            if (!result) {
                // 无法再后退
                console.log("[ModelHandler] Cannot undo further");
                return null;
            }

            console.log("[ModelHandler] Undo successful, generatedCount:", result.generatedCount);
            return await this.convertSessionStep(result);
        } catch (error) {
            console.error("[ModelHandler] Undo error:", error);
            throw error;
        }
    }

    /**
     * 停止生成
     */
    stop(): void {
        if (this.session) {
            this.session.dispose();
            this.session = null;
        }
    }

    /**
     * 释放资源
     */
    dispose(): void {
        if (this.session) {
            this.session.dispose();
            this.session = null;
        }
        this.model = null;
        this.tokenizer = null;
    }

    /**
     * 转换 SessionGenerationStep 为 UI GenerationStep
     */
    private async convertSessionStep(sessionStep: SessionGenerationStep): Promise<GenerationStep> {
        if (!this.tokenizer) {
            throw new Error("Tokenizer not available");
        }

        // Decode top-k token texts
        const topK: TokenCandidate[] = await Promise.all(
            sessionStep.topKTokenIds.map(async (tokenId: number, i: number) => {
                const text = await this.tokenizer!.decode([tokenId], false);
                return {
                    tokenId,
                    text,
                    probability: sessionStep.topKProbs[i],
                };
            })
        );

        // Get selected token text
        const tokenText = await this.tokenizer.decode([sessionStep.tokenId], false);

        // 查找选中 token 的概率
        const selectedProb = topK.find((t) => t.tokenId === sessionStep.tokenId)?.probability ?? 0;

        // 处理 Logit Lens 结果
        let logitLensResults: LayerPrediction[] | undefined;
        // 注意: sessionStep.logitLens 类型来自 @kandle/model-utils
        // 使用 as any 来处理跨包编译时的类型不匹配问题
        const logitLensData = (
            sessionStep as {
                logitLens?: Array<{
                    layerIndex: number;
                    topKTokenIds: number[];
                    topKProbs: Float32Array;
                }>;
            }
        ).logitLens;

        if (logitLensData && logitLensData.length > 0) {
            logitLensResults = await Promise.all(
                logitLensData.map(async (layer) => {
                    // Decode 各层的 top-k token texts
                    const layerTopK: TokenCandidate[] = await Promise.all(
                        layer.topKTokenIds.map(async (tokenId: number, i: number) => {
                            const text = await this.tokenizer!.decode([tokenId], false);
                            return {
                                tokenId,
                                text,
                                probability: layer.topKProbs[i],
                            };
                        })
                    );
                    return {
                        layerIndex: layer.layerIndex,
                        topK: layerTopK,
                    };
                })
            );
        }

        return {
            tokenId: sessionStep.tokenId,
            tokenText,
            probability: selectedProb,
            logProb: sessionStep.logProb,
            topK,
            isEos: sessionStep.isEos,
            cachePosition: sessionStep.cachePosition,
            generatedCount: sessionStep.generatedCount,
            canUndo: sessionStep.canUndo,
            logitLens: logitLensResults,
        };
    }
}
