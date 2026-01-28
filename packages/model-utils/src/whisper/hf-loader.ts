/**
 * HuggingFace Whisper 权重加载工具
 *
 * HuggingFace Whisper 使用分开的 q_proj, k_proj, v_proj 投影，
 * 而 PyTorch 标准 nn.MultiheadAttention 使用合并的 in_proj_weight。
 *
 * 此模块提供权重转换逻辑，将 HF 格式转换为 PyTorch 标准格式。
 *
 * HF 权重结构:
 * - model.encoder.layers.0.self_attn.q_proj.weight [embed_dim, embed_dim]
 * - model.encoder.layers.0.self_attn.k_proj.weight [embed_dim, embed_dim]
 * - model.encoder.layers.0.self_attn.v_proj.weight [embed_dim, embed_dim]
 * - model.encoder.layers.0.self_attn.q_proj.bias [embed_dim]
 * - model.encoder.layers.0.self_attn.k_proj.bias (无 - K 无偏置)
 * - model.encoder.layers.0.self_attn.v_proj.bias [embed_dim]
 * - model.encoder.layers.0.self_attn.out_proj.weight [embed_dim, embed_dim]
 * - model.encoder.layers.0.self_attn.out_proj.bias [embed_dim]
 *
 * PyTorch 标准结构:
 * - encoder.layers.0.self_attn.in_proj_weight [3*embed_dim, embed_dim]
 * - encoder.layers.0.self_attn.in_proj_bias [3*embed_dim]
 * - encoder.layers.0.self_attn.out_proj.weight [embed_dim, embed_dim]
 * - encoder.layers.0.self_attn.out_proj.bias [embed_dim]
 *
 * @module @kandle/model-utils/whisper/hf-loader
 */


import { Tensor, io, cat, zeros } from '@kandle/core';

// ============================================================================
// Types
// ============================================================================

/**
 * 预处理后的权重映射
 * key: 模型参数路径 (如 encoder.layers.0.self_attn.in_proj_weight)
 * value: Tensor 数据
 */
export type ProcessedWeights = Map<string, Tensor>;

/**
 * 权重加载结果
 */
export interface HFWeightLoadResult {
    /** 处理后的权重 */
    weights: ProcessedWeights;

    /** 已处理的 HF 键 (用于追踪) */
    processedHFKeys: Set<string>;

    /** 跳过的 HF 键 (无需转换) */
    skippedHFKeys: Set<string>;
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * 从 SafetensorLayer 读取 Tensor
 */
async function loadTensor(layer: io.SafetensorLayer): Promise<Tensor> {
    return await io.tensorFromSafetensorLayer(layer);
}

/**
 * 合并 Q, K, V 权重为 in_proj_weight
 *
 * @param qWeight - Q 投影权重 [embed_dim, embed_dim]
 * @param kWeight - K 投影权重 [embed_dim, embed_dim]
 * @param vWeight - V 投影权重 [embed_dim, embed_dim]
 * @returns 合并后的 in_proj_weight [3*embed_dim, embed_dim]
 */
async function mergeQKVWeights(
    qWeight: Tensor,
    kWeight: Tensor,
    vWeight: Tensor
): Promise<Tensor> {
    // 使用 cat 沿 dim=0 合并
    // qWeight: [E, E], kWeight: [E, E], vWeight: [E, E]
    // result: [3E, E]
    return cat([qWeight, kWeight, vWeight], 0);
}

/**
 * 合并 Q, K, V 偏置为 in_proj_bias
 *
 * 注意: HF Whisper 的 K 投影没有偏置，需要用零填充
 *
 * @param qBias - Q 投影偏置 [embed_dim]
 * @param vBias - V 投影偏置 [embed_dim]
 * @param embedDim - 嵌入维度
 * @returns 合并后的 in_proj_bias [3*embed_dim]
 */
async function mergeQKVBias(
    qBias: Tensor,
    vBias: Tensor,
    embedDim: number
): Promise<Tensor> {
    // K 没有偏置，用零填充
    const kBias = zeros([embedDim], qBias.dtype);
    return cat([qBias, kBias, vBias], 0);
}

// ============================================================================
// Main Processing Functions
// ============================================================================

/**
 * 判断键是否是 attention 投影键
 */
function isAttnProjKey(key: string): boolean {
    return /\.(self_attn|encoder_attn)\.(q_proj|k_proj|v_proj)\.(weight|bias)$/.test(key);
}

/**
 * 提取 attention 块的基础路径
 * 例如: model.encoder.layers.0.self_attn.q_proj.weight -> model.encoder.layers.0.self_attn
 */
function getAttnBasePath(key: string): string | null {
    const match = key.match(/^(.+\.(self_attn|encoder_attn))\.(q_proj|k_proj|v_proj)\.(weight|bias)$/);
    return match ? match[1] : null;
}

/**
 * 处理 HuggingFace Whisper 权重
 *
 * 将分开的 q_proj, k_proj, v_proj 合并为 in_proj_weight/in_proj_bias
 *
 * @param group - SafetensorGroup
 * @param keyPrefix - 键前缀 (如 'model.' 会被移除)
 * @returns 处理后的权重和元数据
 */
export async function processHFWhisperWeights(
    group: io.SafetensorGroup,
    keyPrefix: string = 'model.'
): Promise<HFWeightLoadResult> {
    const weights: ProcessedWeights = new Map();
    const processedHFKeys = new Set<string>();
    const skippedHFKeys = new Set<string>();

    // 收集所有需要合并的 attention 块
    const attnBlocks = new Map<string, {
        qWeight?: io.SafetensorLayer;
        kWeight?: io.SafetensorLayer;
        vWeight?: io.SafetensorLayer;
        qBias?: io.SafetensorLayer;
        vBias?: io.SafetensorLayer;
    }>();

    // 第一遍：分类所有键
    for (const [hfKey, layer] of group.layers) {
        // 移除前缀
        const modelKey = hfKey.startsWith(keyPrefix)
            ? hfKey.slice(keyPrefix.length)
            : hfKey;

        if (isAttnProjKey(hfKey)) {
            // 是 attention 投影键，需要合并处理
            const basePath = getAttnBasePath(hfKey)!;
            const baseModelPath = basePath.startsWith(keyPrefix)
                ? basePath.slice(keyPrefix.length)
                : basePath;

            if (!attnBlocks.has(baseModelPath)) {
                attnBlocks.set(baseModelPath, {});
            }

            const block = attnBlocks.get(baseModelPath)!;
            if (hfKey.includes('.q_proj.weight')) {
                block.qWeight = layer;
            } else if (hfKey.includes('.k_proj.weight')) {
                block.kWeight = layer;
            } else if (hfKey.includes('.v_proj.weight')) {
                block.vWeight = layer;
            } else if (hfKey.includes('.q_proj.bias')) {
                block.qBias = layer;
            } else if (hfKey.includes('.v_proj.bias')) {
                block.vBias = layer;
            }
            // 注意: k_proj.bias 在 HF Whisper 中不存在

            processedHFKeys.add(hfKey);
        } else {
            // 普通键，直接映射
            skippedHFKeys.add(hfKey);
        }
    }

    // 第二遍：合并 attention 投影权重
    for (const [basePath, block] of attnBlocks) {
        // 合并 weight
        if (block.qWeight && block.kWeight && block.vWeight) {
            const qW = await loadTensor(block.qWeight);
            const kW = await loadTensor(block.kWeight);
            const vW = await loadTensor(block.vWeight);

            const inProjWeight = await mergeQKVWeights(qW, kW, vW);
            weights.set(`${basePath}.in_proj_weight`, inProjWeight);

            // 清理中间张量
            qW.dispose();
            kW.dispose();
            vW.dispose();
        }

        // 合并 bias
        if (block.qBias && block.vBias) {
            const qB = await loadTensor(block.qBias);
            const vB = await loadTensor(block.vBias);
            const embedDim = qB.shape[0];

            const inProjBias = await mergeQKVBias(qB, vB, embedDim);
            weights.set(`${basePath}.in_proj_bias`, inProjBias);

            // 清理中间张量
            qB.dispose();
            vB.dispose();
        }
    }

    return { weights, processedHFKeys, skippedHFKeys };
}

/**
 * 将处理后的权重应用到模型参数
 *
 * @param model - Whisper 模型
 * @param weights - 处理后的权重
 */
export async function applyWeightsToModel(
    model: { namedParameters(prefix?: string, recurse?: boolean): Iterable<[string, any]> },
    weights: ProcessedWeights
): Promise<{ applied: string[]; missing: string[] }> {
    const applied: string[] = [];
    const missing: string[] = [];

    for (const [name, param] of model.namedParameters('', true)) {
        const weight = weights.get(name);
        if (weight) {
            // 复制数据到参数
            const paramData = await param.tensor.dataAsync();
            const weightData = await weight.dataAsync();

            if (paramData.length !== weightData.length) {
                console.warn(
                    `Weight size mismatch for ${name}: ` +
                    `expected ${paramData.length}, got ${weightData.length}`
                );
                missing.push(name);
                continue;
            }

            // 直接复制数据
            (paramData as Float32Array).set(weightData as Float32Array);
            applied.push(name);
        } else {
            missing.push(name);
        }
    }

    return { applied, missing };
}
