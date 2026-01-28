/**
 * Qwen3 模型加载服务
 *
 * 负责：
 * - 初始化 Qwen3ForCausalLM 模型
 * - 加载 safetensor 权重
 * - 初始化 Tokenizer
 */

import { io } from "@kandle/core";
import { Qwen3ForCausalLM, type Qwen3ModelConfig } from "@kandle/model-utils";
import { Tokenizer } from "@huggingface/tokenizers";
import type { ModelFiles } from "../config";

// ============================================================================
// Qwen3-0.6B 配置
// ============================================================================

export const QWEN3_0_6B_CONFIG: Qwen3ModelConfig = {
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
    dtype: "float32",
};

/** EOS (End of Sequence) token IDs */
export const EOS_TOKEN_IDS = [151645, 151643];

// ============================================================================
// Model Loading
// ============================================================================

export interface LoadedQwen3 {
    model: Qwen3ForCausalLM;
    tokenizer: Tokenizer;
}

/**
 * 初始化 Qwen3 模型和 Tokenizer
 *
 * @param files - 已加载的模型文件（tokenizer.json 和 model.safetensors 的 ArrayBuffer）
 * @param onProgress - 进度回调
 * @returns 初始化好的模型和 tokenizer
 */
export async function initQwen3(
    files: ModelFiles,
    onProgress?: (stage: string, progress: number) => void
): Promise<LoadedQwen3> {
    // 1. 初始化 Tokenizer
    onProgress?.("tokenizer", 0);
    console.log("[Qwen3] Initializing tokenizer...");

    if (!files.tokenizer) {
        throw new Error("Tokenizer file is required");
    }

    // Parse tokenizer JSON
    const tokenizerJson = JSON.parse(new TextDecoder().decode(files.tokenizer));
    const tokenizer = new Tokenizer(tokenizerJson, {});

    onProgress?.("tokenizer", 100);
    console.log("[Qwen3] Tokenizer initialized");

    // 2. 创建模型实例
    onProgress?.("model_init", 0);
    console.log("[Qwen3] Creating model instance...");

    const model = new Qwen3ForCausalLM(QWEN3_0_6B_CONFIG, true);

    onProgress?.("model_init", 100);
    console.log("[Qwen3] Model instance created");

    // 3. 加载 safetensor 权重
    onProgress?.("weights", 0);
    console.log("[Qwen3] Loading safetensor weights...");

    const group = await io.loadSafetensor(files.model);
    console.log(`[Qwen3] Safetensor loaded: ${group.layers.size} layers`);

    onProgress?.("weights", 30);

    // 4. 应用权重到模型
    console.log("[Qwen3] Applying weights to model...");

    const result = await model.loadFromSafetensor(group, {
        strict: false, // 允许缺失 rotary_emb.inv_freq 等
        keyMapper: (key: string) => key,
    });

    console.log(
        `[Qwen3] Weights loaded: ${result.loadedKeys.length} loaded, ${result.missingKeys.length} missing, ${result.unexpectedKeys.length} unexpected`
    );

    if (result.missingKeys.length > 0) {
        console.warn("[Qwen3] Missing keys:", result.missingKeys.slice(0, 5));
    }

    onProgress?.("weights", 80);

    // 5. 关闭 group 释放资源
    group.close();

    // 6. 初始化 LM Head
    console.log("[Qwen3] Initializing LM head...");
    model.initLMHead();

    onProgress?.("weights", 100);
    console.log("[Qwen3] Model ready!");

    return { model, tokenizer };
}
