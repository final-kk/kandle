/**
 * useGenerator Hook - 管理 LLM 生成状态
 *
 * 封装 generator 的状态管理，支持：
 * - 单步生成
 * - 自动生成
 * - 暂停/停止
 * - 可操纵生成（干预选择 token）
 * - Mock 模式（用于 UI 开发）
 * - Worker 模式（通过 InferenceWorkerManager）
 */

import { useState, useRef, useCallback, useEffect } from "react";
import type { InferenceWorkerManager } from "../services/worker-manager";
import type {
    GenerationStep as WorkerGenerationStep,
    TokenCandidate as WorkerTokenCandidate,
    LogitLensConfig,
    LayerPrediction as WorkerLayerPrediction,
} from "../workers/message-types";
import type {
    GenerationStep,
    GeneratedToken,
    SamplerConfig,
    GeneratorStatus,
    TokenCandidate,
    LayerPrediction,
} from "../types";
import { DEFAULT_SAMPLER_CONFIG } from "../types";

// ============================================================================
// Hook State
// ============================================================================

export interface UseGeneratorState {
    /** 当前生成状态 */
    status: GeneratorStatus;

    /** 当前步骤信息（包含 top-k 候选） */
    currentStep: GenerationStep | null;

    /** 已生成的 token 列表 */
    generatedTokens: GeneratedToken[];

    /** 完整生成文本 */
    generatedText: string;

    /** 输入 prompt */
    prompt: string;

    /** 采样配置 */
    samplerConfig: SamplerConfig;

    /** Logit Lens 配置 */
    logitLensConfig: LogitLensConfig;

    /** 当前步骤的 Logit Lens 预测结果 */
    currentLogitLens: LayerPrediction[] | null;

    /** 错误信息 */
    error: string | null;

    /** 是否正在执行单步（用于显示 loading） */
    isStepLoading: boolean;

    /** 是否正在自动播放 */
    isAutoPlaying: boolean;

    /** 是否可以后退 */
    canUndo: boolean;
}

export interface UseGeneratorActions {
    /** 开始生成 */
    start: (prompt: string) => Promise<void>;

    /** 单步生成 */
    step: (overrideTokenId?: number) => Promise<void>;

    /** 后退一步 */
    undo: () => Promise<void>;

    /** 自动生成（连续执行直到 EOS 或暂停） */
    autoPlay: (delayMs?: number) => void;

    /** 暂停自动生成 */
    pause: () => void;

    /** 停止生成 */
    stop: () => void;

    /** 重置状态 */
    reset: () => void;

    /** 更新采样配置 */
    setSamplerConfig: (config: Partial<SamplerConfig>) => void;

    /** 更新 Logit Lens 配置 */
    setLogitLensConfig: (config: Partial<LogitLensConfig>) => void;
}

export type UseGeneratorReturn = UseGeneratorState & UseGeneratorActions;

// ============================================================================
// Hook Options
// ============================================================================

export interface UseGeneratorOptions {
    /** 使用 Mock 模式 */
    useMock?: boolean;

    /** Worker Manager 实例（用于真实推理） */
    workerManager?: InferenceWorkerManager | null;

    /** EOS token IDs */
    eosTokenIds?: number[];

    /** 初始 Logit Lens 配置 */
    initialLogitLensConfig?: LogitLensConfig;
}

// ============================================================================
// Default Logit Lens Config
// ============================================================================

const DEFAULT_LOGIT_LENS_CONFIG: LogitLensConfig = {
    enabled: true,
    layerIndices: [0, 5, 11, 16, 22, 27], // 稀疏采样 6 层
    topK: 3,
};

// ============================================================================
// Mock 数据用于 UI 开发
// ============================================================================

const MOCK_VOCAB: Record<number, string> = {
    0: "我",
    1: "你",
    2: "他",
    3: "她",
    4: "它",
    5: "是",
    6: "在",
    7: "的",
    8: "了",
    9: "不",
    10: "有",
    11: "这",
    12: "个",
    13: "人",
    14: "好",
    15: "大",
    16: "小",
    17: "上",
    18: "下",
    19: "中",
    151643: "<|endoftext|>",
};

function generateMockStep(stepIndex: number): GenerationStep {
    // 生成随机 top-k 候选
    const topK: TokenCandidate[] = [];
    let remainingProb = 1.0;
    const usedIds = new Set<number>();

    for (let i = 0; i < 5; i++) {
        // 避免重复
        let tokenId: number;
        do {
            tokenId = Math.floor(Math.random() * 20);
        } while (usedIds.has(tokenId));
        usedIds.add(tokenId);

        // 概率递减
        const prob =
            i === 0 ? 0.3 + Math.random() * 0.3 : remainingProb * (0.4 + Math.random() * 0.3);
        remainingProb -= prob;

        topK.push({
            tokenId,
            text: MOCK_VOCAB[tokenId] || `[${tokenId}]`,
            probability: Math.max(0.01, prob),
        });
    }

    // 按概率排序
    topK.sort((a, b) => b.probability - a.probability);

    // 归一化概率
    const totalProb = topK.reduce((sum, t) => sum + t.probability, 0);
    topK.forEach((t) => (t.probability /= totalProb));

    const selected = topK[0];

    return {
        tokenId: selected.tokenId,
        tokenText: selected.text,
        probability: selected.probability,
        logProb: Math.log(selected.probability),
        topK,
        isEos: stepIndex > 15 && Math.random() > 0.7,
        cachePosition: stepIndex + 10,
        generatedCount: stepIndex + 1,
    };
}

// ============================================================================
// Helper: Convert Worker step to UI step
// ============================================================================

function convertWorkerStep(workerStep: WorkerGenerationStep): GenerationStep {
    const topK: TokenCandidate[] = workerStep.topK.map((c: WorkerTokenCandidate) => ({
        tokenId: c.tokenId,
        text: c.text,
        probability: c.probability,
    }));

    // 转换 Logit Lens 数据
    let logitLens: LayerPrediction[] | undefined;
    if (workerStep.logitLens && workerStep.logitLens.length > 0) {
        logitLens = workerStep.logitLens.map((layer: WorkerLayerPrediction) => ({
            layerIndex: layer.layerIndex,
            topK: layer.topK.map((c: WorkerTokenCandidate) => ({
                tokenId: c.tokenId,
                text: c.text,
                probability: c.probability,
            })),
        }));
    }

    return {
        tokenId: workerStep.tokenId,
        tokenText: workerStep.tokenText,
        probability: workerStep.probability,
        logProb: workerStep.logProb,
        topK,
        isEos: workerStep.isEos,
        cachePosition: workerStep.cachePosition,
        generatedCount: workerStep.generatedCount,
        logitLens,
    };
}

// ============================================================================
// Hook Implementation
// ============================================================================

/**
 * useGenerator Hook
 *
 * @param options - Hook 配置选项
 */
export function useGenerator(options: UseGeneratorOptions | boolean = true): UseGeneratorReturn {
    // 兼容旧的 useMock 布尔参数
    const opts: UseGeneratorOptions = typeof options === "boolean" ? { useMock: options } : options;

    const {
        useMock = true,
        workerManager = null,
        eosTokenIds = [151643, 151645], // Qwen3 EOS tokens
        initialLogitLensConfig,
    } = opts;

    // 状态
    const [status, setStatus] = useState<GeneratorStatus>("idle");
    const [currentStep, setCurrentStep] = useState<GenerationStep | null>(null);
    const [generatedTokens, setGeneratedTokens] = useState<GeneratedToken[]>([]);
    const [prompt, setPrompt] = useState<string>("");
    const [samplerConfig, setSamplerConfigState] = useState<SamplerConfig>(DEFAULT_SAMPLER_CONFIG);
    const [logitLensConfig, setLogitLensConfigState] = useState<LogitLensConfig>(
        initialLogitLensConfig ?? DEFAULT_LOGIT_LENS_CONFIG
    );
    const [currentLogitLens, setCurrentLogitLens] = useState<LayerPrediction[] | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [isStepLoading, setIsStepLoading] = useState<boolean>(false);
    const [isAutoPlaying, setIsAutoPlaying] = useState<boolean>(false);

    // Refs
    const autoPlayRef = useRef<boolean>(false);
    const stepIndexRef = useRef<number>(0);

    // 计算生成文本
    const generatedText = generatedTokens.map((t) => t.text).join("");

    // ========================================
    // Worker Event Subscriptions
    // ========================================

    useEffect(() => {
        if (!workerManager || useMock) {
            return;
        }

        // Subscribe to generation events
        const unsubStep = workerManager.on("generationStep", (step) => {
            const uiStep = convertWorkerStep(step);
            setCurrentStep(uiStep);
            // 更新 Logit Lens 数据
            if (uiStep.logitLens) {
                setCurrentLogitLens(uiStep.logitLens);
            }
            setIsStepLoading(false);
            stepIndexRef.current++;
        });

        const unsubComplete = workerManager.on("generationComplete", () => {
            setStatus("complete");
            setCurrentStep(null);
            setCurrentLogitLens(null);
            setIsStepLoading(false);
            autoPlayRef.current = false;
        });

        const unsubError = workerManager.on("error", (err) => {
            console.error("[useGenerator] Worker error:", err);
            setError(err.message);
            setStatus("error");
            setIsStepLoading(false);
            autoPlayRef.current = false;
        });

        const unsubStarted = workerManager.on("generationStarted", () => {
            setStatus("generating");
        });

        // 订阅 undo 事件
        const unsubUndoComplete = workerManager.on("undoComplete", (step) => {
            const uiStep = convertWorkerStep(step);
            setCurrentStep(uiStep);
            // 更新 Logit Lens 数据
            if (uiStep.logitLens) {
                setCurrentLogitLens(uiStep.logitLens);
            }
            setIsStepLoading(false);
            // generatedTokens 在 undo action 中已更新
        });

        const unsubUndoFailed = workerManager.on("undoFailed", (payload) => {
            console.warn("[useGenerator] Undo failed:", payload.reason);
            setIsStepLoading(false);
        });

        return () => {
            unsubStep();
            unsubComplete();
            unsubError();
            unsubStarted();
            unsubUndoComplete();
            unsubUndoFailed();
        };
    }, [workerManager, useMock]);

    // ========================================
    // Actions
    // ========================================

    const reset = useCallback(() => {
        setStatus("idle");
        setCurrentStep(null);
        setGeneratedTokens([]);
        setPrompt("");
        setError(null);
        setCurrentLogitLens(null);
        autoPlayRef.current = false;
        stepIndexRef.current = 0;
    }, []);

    const start = useCallback(
        async (inputPrompt: string) => {
            reset();
            setPrompt(inputPrompt);
            setStatus("generating");

            if (useMock) {
                // Mock: 生成第一步
                const step = generateMockStep(0);
                setCurrentStep(step);
                stepIndexRef.current = 1;
            } else if (workerManager) {
                try {
                    // 通过 Worker 开始生成
                    workerManager.startGeneration({
                        prompt: inputPrompt,
                        temperature: samplerConfig.temperature,
                        topK: samplerConfig.topK,
                        topP: samplerConfig.topP,
                        doSample: samplerConfig.doSample,
                        maxNewTokens: 512,
                        displayTopK: 10,
                        eosTokenIds,
                        // Logit Lens 配置
                        logitLens: logitLensConfig.enabled ? logitLensConfig : undefined,
                    });
                    // Worker 会异步发送 generationStep 事件
                } catch (e) {
                    const errorMsg = (e as Error).message;
                    console.error("[useGenerator] start error:", e);
                    setError(errorMsg);
                    setStatus("error");
                }
            } else {
                setError("Worker Manager 未初始化");
                setStatus("error");
            }
        },
        [useMock, workerManager, samplerConfig, logitLensConfig, eosTokenIds, reset]
    );

    const step = useCallback(
        async (overrideTokenId?: number) => {
            if (status !== "generating" && status !== "paused") {
                return;
            }

            if (!currentStep) {
                return;
            }

            // 开始 loading
            setIsStepLoading(true);

            // 确定使用的 token
            const tokenId = overrideTokenId ?? currentStep.tokenId;
            const isOverride =
                overrideTokenId !== undefined && overrideTokenId !== currentStep.tokenId;

            // 查找 token 信息
            let tokenInfo = currentStep.topK.find((t) => t.tokenId === tokenId);
            if (!tokenInfo) {
                tokenInfo = {
                    tokenId,
                    text: useMock ? MOCK_VOCAB[tokenId] || `[${tokenId}]` : `[${tokenId}]`,
                    probability: 0.01,
                };
            }

            // 添加到已生成列表
            const newToken: GeneratedToken = {
                id: tokenId,
                text: tokenInfo.text,
                probability: tokenInfo.probability,
                isOverride,
            };
            setGeneratedTokens((prev) => [...prev, newToken]);

            // 检查 EOS
            if (currentStep.isEos || eosTokenIds.includes(tokenId)) {
                setStatus("complete");
                setCurrentStep(null);
                setIsStepLoading(false);
                return;
            }

            if (useMock) {
                // Mock: 模拟延迟
                await new Promise((resolve) => setTimeout(resolve, 100));
                // Mock: 生成下一步
                const nextStep = generateMockStep(stepIndexRef.current);
                setCurrentStep(nextStep);
                stepIndexRef.current++;
                setStatus("generating");
                setIsStepLoading(false);
            } else if (workerManager) {
                // 通过 Worker 执行单步
                // Worker 会异步发送 generationStep 事件
                workerManager.step(overrideTokenId);
                setStatus("generating");
                // isStepLoading 会在收到 generationStep 事件时设为 false
            }
        },
        [status, currentStep, useMock, workerManager, eosTokenIds]
    );

    const autoPlay = useCallback(
        (delayMs: number = 200) => {
            if (status !== "generating" && status !== "paused") {
                return;
            }

            autoPlayRef.current = true;
            setIsAutoPlaying(true);
            setStatus("generating");

            const runStep = async () => {
                if (!autoPlayRef.current) {
                    setIsAutoPlaying(false);
                    return;
                }

                await step();

                // 继续下一步 - 需要重新检查状态
                if (autoPlayRef.current) {
                    setTimeout(runStep, delayMs);
                } else {
                    setIsAutoPlaying(false);
                }
            };

            runStep();
        },
        [status, step]
    );

    const pause = useCallback(() => {
        autoPlayRef.current = false;
        setIsAutoPlaying(false);
        if (status === "generating") {
            setStatus("paused");
        }
    }, [status]);

    const undo = useCallback(async () => {
        // 不能在自动播放时后退
        if (isAutoPlaying) {
            return;
        }

        // 必须有已生成的 token 才能后退
        if (generatedTokens.length === 0) {
            return;
        }

        // 开始 loading
        setIsStepLoading(true);

        if (useMock) {
            // Mock: 模拟后退
            setGeneratedTokens((prev) => prev.slice(0, -1));
            stepIndexRef.current = Math.max(0, stepIndexRef.current - 1);

            // Mock: 模拟延迟
            await new Promise((resolve) => setTimeout(resolve, 100));

            // Mock: 重新生成该位置的预测
            const newStep = generateMockStep(stepIndexRef.current);
            setCurrentStep(newStep);
            setIsStepLoading(false);

            // 如果之前是 complete 状态，回退后应该恢复为 generating
            if (status === "complete") {
                setStatus("generating");
            }
        } else if (workerManager) {
            // 先更新 UI 状态
            setGeneratedTokens((prev) => prev.slice(0, -1));
            stepIndexRef.current = Math.max(0, stepIndexRef.current - 1);

            // 如果之前是 complete 状态，回退后应该恢复为 generating
            if (status === "complete") {
                setStatus("generating");
            }

            // 通过 Worker 执行后退
            // Worker 会异步发送 undoComplete 事件
            workerManager.undo();
        }
    }, [isAutoPlaying, generatedTokens.length, useMock, workerManager, status]);

    const stop = useCallback(() => {
        autoPlayRef.current = false;
        if (workerManager && !useMock) {
            workerManager.stop();
        }
        setStatus("complete");
        setCurrentStep(null);
    }, [workerManager, useMock]);

    const setSamplerConfig = useCallback((config: Partial<SamplerConfig>) => {
        setSamplerConfigState((prev) => ({ ...prev, ...config }));
    }, []);

    const setLogitLensConfig = useCallback((config: Partial<LogitLensConfig>) => {
        setLogitLensConfigState((prev) => ({ ...prev, ...config }));
    }, []);

    // 计算 canUndo
    const canUndo = generatedTokens.length > 0 && !isAutoPlaying && !isStepLoading;

    return {
        // State
        status,
        currentStep,
        generatedTokens,
        generatedText,
        prompt,
        samplerConfig,
        logitLensConfig,
        currentLogitLens,
        error,
        isStepLoading,
        isAutoPlaying,
        canUndo,

        // Actions
        start,
        step,
        undo,
        autoPlay,
        pause,
        stop,
        reset,
        setSamplerConfig,
        setLogitLensConfig,
    };
}
