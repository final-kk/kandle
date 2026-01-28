/**
 * InterpretabilityLayout - 可解释性 Demo 主布局
 *
 * 布局结构（3 列）：
 * ┌─────────────────────────────────────────────────────────────────────────────┐
 * │ Header                                                                       │
 * ├─────────────────────────────────────────────────────────────────────────────┤
 * │ Eager Mode Description (inline bar)                                          │
 * ├────────────────┬──────────────────────────────────────┬─────────────────────┤
 * │  Left Panel    │  Main Workspace                      │  Right Panel        │
 * │  - Model Load  │  ┌────────────────────────────────┐  │  ┌───────────────┐  │
 * │  - Prompt      │  │ Token Probs HUD                │  │  │ Gen Controls  │  │
 * │  - Sampler     │  └────────────────────────────────┘  │  └───────────────┘  │
 * │                │  ┌────────────────────────────────┐  │  ┌───────────────┐  │
 * │                │  │ Text Stream                    │  │  │ Logit Lens    │  │
 * │                │  └────────────────────────────────┘  │  │ (Config +     │  │
 * │                │  ┌────────────────────────────────┐  │  │  Panel)       │  │
 * │                │  │ Attention Vis                  │  │  └───────────────┘  │
 * │                │  └────────────────────────────────┘  │                     │
 * └────────────────┴──────────────────────────────────────┴─────────────────────┘
 */

import { useState, useCallback, useEffect } from "react";
import { useGenerator } from "../../hooks/useGenerator";

// UI Components
import { ModelLoader } from "../ui/ModelLoader";

// Controls
import { GenerationControls } from "../controls/GenerationControls";
import { SamplerSettings } from "../controls/SamplerSettings";

// Visualizations
import { TokenProbsHUD } from "./TokenProbsHUD";
import { TextStream } from "./TextStream";
import { AttentionLinks } from "./AttentionLinks";
import { LogitLensPanel } from "./LogitLensPanel";
import { LogitLensConfig } from "./LogitLensConfig";

// Types
import type { LoadProgress } from "../../workers/message-types";
import type { LoadMethod } from "../../services/model-loader";
import type { InferenceWorkerManager } from "../../services/worker-manager";
import type { GeneratedToken } from "../../types";

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * 生成 Mock Attention 数据用于 UI 开发
 * 真实模型集成后将替换为实际的 attention 权重
 */
function generateMockAttention(prompt: string, generatedTokens: GeneratedToken[]): number[][] {
    // 简化 token 列表
    const promptTokens = prompt.split("").slice(0, 10);
    const genTokens = generatedTokens.map((t) => t.text);
    const totalTokens = promptTokens.length + genTokens.length;

    if (totalTokens === 0) return [];

    // 生成随机 attention 矩阵
    const weights: number[][] = [];

    for (let i = 0; i < totalTokens; i++) {
        const row: number[] = [];
        let sum = 0;

        for (let j = 0; j < totalTokens; j++) {
            // 只关注之前的 token (causal mask)
            if (j > i) {
                row.push(0);
            } else {
                // 距离越近，attention 越高
                const distance = i - j;
                const base = Math.exp(-distance * 0.3);
                const noise = Math.random() * 0.2;
                const weight = base + noise;
                row.push(weight);
                sum += weight;
            }
        }

        // 归一化
        weights.push(row.map((w) => (sum > 0 ? w / sum : 0)));
    }

    return weights;
}

// ============================================================================
// Types
// ============================================================================

export interface InterpretabilityLayoutProps {
    /** 是否使用 Mock 数据 */
    useMock?: boolean;

    /** WebGPU 状态 */
    webgpuAvailable?: boolean;

    /** 模型是否已加载 */
    modelLoaded: boolean;

    /** 模型是否正在加载 */
    isLoadingModel: boolean;

    /** 加载进度 */
    loadProgress: LoadProgress | null;

    /** 加载错误 */
    loadError: string | null;

    /** 模型名称 */
    modelName?: string;

    /** 加载模型回调 */
    onLoadModel: (
        files: FileList | null,
        customUrls?: { tokenizer?: string; model?: string }
    ) => void;

    /** 模型 URL 配置 */
    modelUrls: { tokenizer: string; model: string };

    /** Worker Manager 实例（用于真实推理） */
    workerManager?: InferenceWorkerManager | null;
}

// ============================================================================
// Component
// ============================================================================

export function InterpretabilityLayout({
    useMock = true,
    webgpuAvailable = false,
    modelLoaded,
    isLoadingModel,
    loadProgress,
    loadError,
    modelName = "Qwen3-0.6B",
    onLoadModel,
    modelUrls,
    workerManager = null,
}: InterpretabilityLayoutProps) {
    // Generator state - use workerManager when available
    const generator = useGenerator({
        useMock: useMock || !workerManager?.modelLoaded,
        workerManager,
    });

    // Local state
    const [promptInput, setPromptInput] = useState("");
    const [loadMethod, setLoadMethod] = useState<LoadMethod>("url");
    const [showLogitLensConfig, setShowLogitLensConfig] = useState(false);
    const [showModelLoader, setShowModelLoader] = useState(!modelLoaded);

    // 模型加载完成后自动隐藏 ModelLoader
    useEffect(() => {
        if (modelLoaded) {
            setShowModelLoader(false);
        }
    }, [modelLoaded]);

    // Handlers
    const handleStart = useCallback(() => {
        if (promptInput.trim()) {
            generator.start(promptInput.trim());
        }
    }, [promptInput, generator]);

    const handleStep = useCallback(() => {
        generator.step();
    }, [generator]);

    const handleTokenSelect = useCallback(
        (tokenId: number) => {
            generator.step(tokenId);
        },
        [generator]
    );

    const handleKeyDown = useCallback(
        (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleStart();
            }
        },
        [handleStart]
    );

    const isRunning = generator.status === "generating" || generator.status === "paused";
    const canStartGeneration = modelLoaded || useMock;

    return (
        <div className="min-h-screen bg-grid flex flex-col">
            {/* Header */}
            <header className="border-b border-gray-800 bg-gray-950/80 backdrop-blur-sm sticky top-0 z-50">
                <div className="container mx-auto px-4 py-3 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <h1 className="text-2xl font-bold text-gradient-cyber">KANDLE</h1>
                        <span className="text-xs text-gray-500 font-mono">
                            Interpretability Demo
                        </span>
                    </div>
                    <div className="flex items-center gap-4">
                        {modelLoaded && (
                            <span className="text-xs font-mono text-cyber-green">
                                {modelName} loaded
                            </span>
                        )}
                        <div
                            className={`w-3 h-3 rounded-full ${
                                webgpuAvailable ? "bg-cyber-green animate-pulse" : "bg-red-500"
                            }`}
                            title={webgpuAvailable ? "WebGPU Ready" : "WebGPU Unavailable"}
                        />
                    </div>
                </div>
            </header>

            {/* Eager Mode Description */}
            <div className="container mx-auto px-4 py-3">
                <div className="flex items-center gap-6 px-4 py-2 rounded-lg bg-gray-900/50 border border-cyber-purple/30">
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-cyber-purple animate-pulse" />
                        <span className="text-cyber-purple font-medium text-sm">Eager Mode</span>
                    </div>
                    <div className="flex items-center gap-4 text-xs text-gray-400">
                        <span className="flex items-center gap-1">
                            <span className="text-gray-600">|</span>
                            Step-by-step prediction
                        </span>
                        <span className="flex items-center gap-1">
                            <span className="text-gray-600">|</span>
                            Click to override token
                        </span>
                        <span className="flex items-center gap-1">
                            <span className="text-gray-600">|</span>
                            Real-time attention weights
                        </span>
                        <span className="flex items-center gap-1">
                            <span className="text-gray-600">|</span>
                            Sampling params take effect instantly
                        </span>
                    </div>
                </div>
            </div>

            {/* Main Content - 3 Column Layout */}
            <main className="flex-1 container mx-auto px-4 py-4">
                <div className="grid grid-cols-12 gap-4 h-full">
                    {/* Left Panel - Controls (3 cols) */}
                    <aside className="col-span-12 lg:col-span-3 space-y-4">
                        {/* Model Loader */}
                        <div className="panel">
                            <div className="panel-header flex items-center justify-between">
                                <span className="text-cyber-cyan">Load Model</span>
                                <div className="flex items-center gap-2">
                                    {modelLoaded && (
                                        <span className="text-xs text-cyber-green">Ready</span>
                                    )}
                                    <button
                                        onClick={() => setShowModelLoader(!showModelLoader)}
                                        className="text-xs text-gray-400 hover:text-white"
                                    >
                                        {showModelLoader ? "Hide" : "Show"}
                                    </button>
                                </div>
                            </div>
                            {showModelLoader && (
                                <div className="p-3">
                                    <ModelLoader
                                        modelType="qwen3"
                                        loadMethod={loadMethod}
                                        onLoadMethodChange={setLoadMethod}
                                        onLoad={onLoadModel}
                                        isLoading={isLoadingModel}
                                        progress={loadProgress}
                                        error={loadError}
                                        defaultUrls={modelUrls}
                                    />
                                </div>
                            )}
                        </div>

                        {/* Prompt Input */}
                        <div className="panel">
                            <div className="panel-header">
                                <span className="text-cyber-cyan">Prompt</span>
                            </div>
                            <div className="p-3">
                                <textarea
                                    value={promptInput}
                                    onChange={(e) => setPromptInput(e.target.value)}
                                    onKeyDown={handleKeyDown}
                                    placeholder={
                                        canStartGeneration
                                            ? "Enter your prompt here..."
                                            : "Load a model first..."
                                    }
                                    disabled={isRunning || !canStartGeneration}
                                    rows={4}
                                    className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white text-sm font-mono resize-none focus:outline-none focus:border-cyber-cyan disabled:opacity-50"
                                />
                                <button
                                    onClick={handleStart}
                                    disabled={
                                        !promptInput.trim() || isRunning || !canStartGeneration
                                    }
                                    className="w-full mt-2 btn-cyber btn-cyber-primary"
                                >
                                    {!canStartGeneration ? "Load Model First" : "Start Generation"}
                                </button>
                            </div>
                        </div>

                        {/* Sampler Settings */}
                        <SamplerSettings
                            config={generator.samplerConfig}
                            onChange={generator.setSamplerConfig}
                            disabled={isRunning}
                        />
                    </aside>

                    {/* Main Workspace (6 cols) */}
                    <section className="col-span-12 lg:col-span-6 space-y-4">
                        {/* Token Probs HUD */}
                        <TokenProbsHUD
                            topK={generator.currentStep?.topK ?? []}
                            selectedTokenId={generator.currentStep?.tokenId ?? null}
                            onTokenSelect={handleTokenSelect}
                            isGenerating={generator.status === "generating"}
                            interactive={
                                !generator.isAutoPlaying &&
                                (generator.status === "generating" || generator.status === "paused")
                            }
                            isLoading={generator.isStepLoading}
                        />

                        {/* Text Stream */}
                        <TextStream
                            prompt={generator.prompt}
                            generatedTokens={generator.generatedTokens}
                            isGenerating={generator.status === "generating"}
                            showDetails={true}
                            maxHeight={300}
                        />

                        {/* Attention Visualization - 始终显示 */}
                        <AttentionLinks
                            data={
                                generator.generatedTokens.length > 0
                                    ? {
                                          // 暂时使用 mock 数据，等待真实模型集成
                                          weights: generateMockAttention(
                                              generator.prompt,
                                              generator.generatedTokens
                                          ),
                                          layerIndex: 0,
                                          tokens: [
                                              ...generator.prompt.split("").slice(0, 10),
                                              ...generator.generatedTokens.map((t) => t.text),
                                          ],
                                      }
                                    : null
                            }
                            numLayers={28}
                            maxHeight={280}
                        />

                        {/* Error Display */}
                        {generator.error && (
                            <div className="panel border-red-500/50">
                                <div className="panel-header text-red-400">Error</div>
                                <div className="p-3 text-sm text-red-300">{generator.error}</div>
                            </div>
                        )}
                    </section>

                    {/* Right Panel - Controls + Logit Lens (3 cols) */}
                    <aside className="col-span-12 lg:col-span-3 flex flex-col gap-4 h-[calc(100vh-220px)]">
                        {/* Generation Controls */}
                        <GenerationControls
                            status={generator.status}
                            onStep={handleStep}
                            onUndo={generator.undo}
                            canUndo={generator.canUndo}
                            onAutoPlay={() => generator.autoPlay(300)}
                            onPause={generator.pause}
                            onStop={generator.stop}
                            onReset={generator.reset}
                            disabled={!isRunning && generator.status !== "complete"}
                        />

                        {/* Combined Logit Lens Panel */}
                        <div className="panel flex-1 flex flex-col min-h-0">
                            {/* Header with config toggle */}
                            <div className="panel-header flex items-center justify-between flex-shrink-0">
                                <span className="text-cyber-purple">Logit Lens</span>
                                <button
                                    onClick={() => setShowLogitLensConfig(!showLogitLensConfig)}
                                    className="text-xs text-gray-400 hover:text-white"
                                >
                                    {showLogitLensConfig ? "Hide Config" : "Show Config"}
                                </button>
                            </div>

                            {/* Collapsible Config Section */}
                            {showLogitLensConfig && (
                                <div className="p-3 border-b border-gray-700/50 flex-shrink-0">
                                    <LogitLensConfig
                                        config={generator.logitLensConfig}
                                        onChange={generator.setLogitLensConfig}
                                        disabled={isRunning}
                                        totalLayers={28}
                                    />
                                </div>
                            )}

                            {/* Logit Lens Panel Content */}
                            <div className="flex-1 min-h-0 overflow-hidden">
                                <LogitLensPanel
                                    layerPredictions={generator.currentLogitLens}
                                    isGenerating={generator.status === "generating"}
                                    isLoading={generator.isStepLoading}
                                    totalLayers={28}
                                />
                            </div>
                        </div>
                    </aside>
                </div>
            </main>

            {/* Footer */}
            <footer className="border-t border-gray-800 py-4">
                <div className="container mx-auto px-4 text-center text-xs text-gray-600">
                    <p>Kandle - WebGPU Eager Mode AI Framework | Interpretability Demo</p>
                </div>
            </footer>
        </div>
    );
}
