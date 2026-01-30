/**
 * Worker 通信协议类型定义
 *
 * 定义 Main Thread <-> Inference Worker 之间的所有消息类型
 */

// ============================================================================
// Commands (Main Thread → Worker)
// ============================================================================

export type WorkerCommand =
    | { type: "init" }
    | { type: "loadModel"; payload: LoadModelPayload }
    | { type: "loadModelFromBuffer"; payload: LoadModelFromBufferPayload }
    | { type: "startGeneration"; payload: StartGenerationPayload }
    | { type: "step"; payload?: StepPayload }
    | { type: "undo" }
    | { type: "stop" }
    | { type: "dispose" };

/** 模型加载配置（URL 模式） */
export interface LoadModelPayload {
    tokenizerUrl: string;
    modelUrl: string;
}

/** 模型加载配置（Buffer 模式，用于文件上传） */
export interface LoadModelFromBufferPayload {
    tokenizerBuffer: ArrayBuffer;
    modelBuffer: ArrayBuffer;
}

/** 生成配置 */
export interface StartGenerationPayload {
    prompt: string;
    temperature: number;
    topK: number;
    topP: number;
    doSample: boolean;
    maxNewTokens: number;
    displayTopK: number;
    eosTokenIds: number[];
    /** Logit Lens 配置 */
    logitLens?: LogitLensConfig;
    /** Attention 可视化配置 */
    attention?: AttentionConfig;
}

/** Logit Lens 配置 */
export interface LogitLensConfig {
    /** 是否启用 */
    enabled: boolean;
    /** 需要收集的层索引 */
    layerIndices: number[];
    /** 每层返回的 top-k 数量 */
    topK: number;
}

/** Attention 可视化配置 */
export interface AttentionConfig {
    /** 是否启用 */
    enabled: boolean;
    /** 需要捕获 attention weights 的层索引 */
    layerIndices: number[];
}

/** 单步生成配置 */
export interface StepPayload {
    overrideTokenId?: number;
}

// ============================================================================
// Events (Worker → Main Thread)
// ============================================================================

export type WorkerEvent =
    | { type: "ready"; payload: WebGPUInfo }
    | { type: "loadProgress"; payload: LoadProgress }
    | { type: "modelReady" }
    | { type: "generationStarted" }
    | { type: "generationStep"; payload: GenerationStep }
    | { type: "generationComplete" }
    | { type: "undoComplete"; payload: GenerationStep }
    | { type: "undoFailed"; payload: { reason: string } }
    | { type: "error"; payload: ErrorPayload };

/** WebGPU 信息 */
export interface WebGPUInfo {
    available: boolean;
    supportsF16: boolean;
    adapterInfo?: string;
}

/** 加载进度 */
export interface LoadProgress {
    stage: "tokenizer" | "model" | "weights";
    loaded: number;
    total: number;
    fileName: string;
    speed?: number;
}

/** 生成步骤结果 */
export interface GenerationStep {
    /** 选中的 token ID */
    tokenId: number;
    /** 选中 token 的文本 */
    tokenText: string;
    /** 选中 token 的概率 */
    probability: number;
    /** 选中 token 的 log 概率 */
    logProb: number;
    /** Top-K 候选列表 */
    topK: TokenCandidate[];
    /** 是否为结束 token */
    isEos: boolean;
    /** 当前 KV Cache 位置 */
    cachePosition: number;
    /** 已生成的 token 总数 */
    generatedCount: number;
    /** 是否可以后退 */
    canUndo: boolean;
    /** Logit Lens 各层预测结果 */
    logitLens?: LayerPrediction[];
    /** 各层 Attention Weights 数据 */
    attentionData?: LayerAttentionData[];
}

/**
 * 单层的 Attention Weights 数据
 * 设计为 Transferable 友好格式
 */
export interface LayerAttentionData {
    /** 层索引 */
    layerIndex: number;
    /**
     * Attention weights 数据 (Float32Array)
     * 形状: [numHeads, querySeqLen, keySeqLen]
     * 扁平化存储，使用 Transferable 传输
     */
    weights: Float32Array;
    /** 头数量 */
    numHeads: number;
    /** Query 序列长度 */
    querySeqLen: number;
    /** Key 序列长度 */
    keySeqLen: number;
}

/** 单层的 Logit Lens 预测结果 */
export interface LayerPrediction {
    /** 层索引 */
    layerIndex: number;
    /** Top-K 候选列表 */
    topK: TokenCandidate[];
}

/** Token 候选 */
export interface TokenCandidate {
    tokenId: number;
    text: string;
    probability: number;
}

/** 错误信息 */
export interface ErrorPayload {
    /** 错误发生的操作 */
    operation: string;
    /** 错误消息 */
    message: string;
    /** 错误堆栈（可选） */
    stack?: string;
}
