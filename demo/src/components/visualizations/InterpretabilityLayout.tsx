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
 * 从已生成 token 和 attention 数据创建 token 文本数组
 * 用于 attention 可视化的 x/y 轴标签
 *
 * @param generatedTokens - 已生成的 token 列表（包含文本）
 * @param keySeqLen - attention 数据中的 key 序列长度（总 token 数）
 */
function buildTokenLabels(generatedTokens: GeneratedToken[], keySeqLen: number): string[] {
    // generatedTokens 只包含生成的 token，不包含 prompt tokens
    // keySeqLen = promptLength + generatedCount
    const promptLength = keySeqLen - generatedTokens.length;

    // 为 prompt tokens 创建占位符标签
    const promptLabels = Array.from({ length: promptLength }, (_, i) => `[${i}]`);

    // 使用真实的生成 token 文本
    const genLabels = generatedTokens.map((t) => t.text);

    return [...promptLabels, ...genLabels];
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

                        <a
                            href="https://github.com/final-kk/kandle"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-gray-400 hover:text-white transition-colors"
                            title="View on GitHub"
                        >
                            <svg height="20" width="20" viewBox="0 0 16 16" fill="currentColor">
                                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
                            </svg>
                        </a>

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

                        {/* Attention Visualization - 使用真实数据 */}
                        <AttentionLinks
                            data={generator.currentAttention ?? null}
                            tokens={
                                generator.currentAttention && generator.currentAttention.length > 0
                                    ? buildTokenLabels(
                                          generator.generatedTokens,
                                          generator.currentAttention[0].keySeqLen
                                      )
                                    : []
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
