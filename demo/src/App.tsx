/**
 * Kandle Interpretability Demo
 *
 * A demonstration of LLM interpretability features using WebGPU eager mode execution.
 * This demo showcases:
 * - Step-by-step token generation
 * - Top-K probability visualization
 * - Interventional generation (click to override token selection)
 * - Real-time attention weight capture
 *
 * 使用 Web Worker 架构：所有 WebGPU 操作在 Worker 中执行，主线程保持流畅
 */

import { useState, useEffect, useCallback } from "react";
import { InterpretabilityLayout } from "./components/visualizations";
import { getWorkerManager, type InferenceWorkerManager } from "./services/worker-manager";
import type { WebGPUInfo, LoadProgress } from "./workers/message-types";
import { QWEN3_MODEL_URLS } from "./config";

// ============================================================================
// App Component
// ============================================================================

function App() {
    // Worker Manager 状态
    const [workerManager, setWorkerManager] = useState<InferenceWorkerManager | null>(null);

    // Backend state
    const [webgpuInfo, setWebgpuInfo] = useState<WebGPUInfo | null>(null);
    const [backendError, setBackendError] = useState<string | null>(null);
    const [isInitializing, setIsInitializing] = useState(true);

    // Model state
    const [modelLoaded, setModelLoaded] = useState(false);
    const [isLoadingModel, setIsLoadingModel] = useState(false);
    const [loadProgress, setLoadProgress] = useState<LoadProgress | null>(null);
    const [loadError, setLoadError] = useState<string | null>(null);

    // Initialize Worker and WebGPU backend
    useEffect(() => {
        let mounted = true;

        const init = async () => {
            console.log("[Kandle] Initializing Worker...");
            try {
                const manager = getWorkerManager();

                // Initialize WebGPU in Worker
                const info = await manager.init();

                if (mounted) {
                    setWorkerManager(manager);
                    setWebgpuInfo(info);
                    console.log("[Kandle] Worker initialized successfully:", info);
                    setIsInitializing(false);
                }
            } catch (e) {
                if (mounted) {
                    const msg = (e as Error).message;
                    setBackendError(msg);
                    console.error("[Kandle] Worker initialization failed:", msg);
                    setIsInitializing(false);
                }
            }
        };

        init();

        return () => {
            mounted = false;
            // Note: 不在这里 dispose worker，因为是单例模式
        };
    }, []);

    // Handle model loading - 使用 WorkerManager 在 Worker 中加载
    const handleLoadModel = useCallback(
        async (files: FileList | null, customUrls?: { tokenizer?: string; model?: string }) => {
            if (!workerManager) {
                setLoadError("Worker not initialized");
                return;
            }

            setIsLoadingModel(true);
            setLoadError(null);
            setLoadProgress(null);
            setModelLoaded(false);

            try {
                if (files && files.length > 0) {
                    // 文件上传模式（Upload 或 WebFile API）
                    console.log("[Kandle] Loading model from files...");

                    let tokenizerBuffer: ArrayBuffer | null = null;
                    let modelBuffer: ArrayBuffer | null = null;

                    // 解析文件列表
                    for (let i = 0; i < files.length; i++) {
                        const file = files[i];
                        const fileName = file.name.toLowerCase();

                        setLoadProgress({
                            stage: fileName.endsWith(".json") ? "tokenizer" : "model",
                            loaded: 0,
                            total: file.size,
                            fileName: file.name,
                        });

                        if (fileName.endsWith(".json")) {
                            console.log("[Kandle] Reading tokenizer file:", file.name);
                            tokenizerBuffer = await file.arrayBuffer();
                        } else if (fileName.endsWith(".safetensors")) {
                            console.log("[Kandle] Reading model file:", file.name);
                            modelBuffer = await file.arrayBuffer();
                        }
                    }

                    if (!tokenizerBuffer) {
                        throw new Error("No tokenizer.json file found in selection");
                    }
                    if (!modelBuffer) {
                        throw new Error("No model.safetensors file found in selection");
                    }

                    console.log(
                        `[Kandle] Files read: tokenizer=${tokenizerBuffer.byteLength}, model=${modelBuffer.byteLength}`
                    );

                    // 通过 Worker 加载（使用 Buffer 模式）
                    await workerManager.loadModelFromBuffer(
                        tokenizerBuffer,
                        modelBuffer,
                        (progress) => {
                            setLoadProgress(progress);
                        }
                    );
                } else if (customUrls?.tokenizer && customUrls?.model) {
                    // URL 模式
                    console.log("[Kandle] Loading model via URL...");
                    console.log("[Kandle] Tokenizer URL:", customUrls.tokenizer);
                    console.log("[Kandle] Model URL:", customUrls.model);

                    await workerManager.loadModel(
                        customUrls.tokenizer,
                        customUrls.model,
                        (progress) => {
                            setLoadProgress(progress);
                        }
                    );
                } else {
                    // WebFile API 模式（无 files 参数时触发文件选择器）
                    console.log("[Kandle] Opening file picker (WebFile API)...");

                    // 检查 API 是否可用
                    if (!("showOpenFilePicker" in window)) {
                        throw new Error(
                            "File System Access API not supported in this browser. Please use URL or Upload mode."
                        );
                    }

                    // 打开文件选择器
                    const fileHandles = await (
                        window as unknown as {
                            showOpenFilePicker: (options: {
                                multiple: boolean;
                                types: Array<{
                                    description: string;
                                    accept: Record<string, string[]>;
                                }>;
                            }) => Promise<FileSystemFileHandle[]>;
                        }
                    ).showOpenFilePicker({
                        multiple: true,
                        types: [
                            {
                                description: "Model Files",
                                accept: {
                                    "application/octet-stream": [".safetensors", ".json"],
                                },
                            },
                        ],
                    });

                    let tokenizerBuffer: ArrayBuffer | null = null;
                    let modelBuffer: ArrayBuffer | null = null;

                    for (const handle of fileHandles) {
                        const file: File = await handle.getFile();
                        const fileName = file.name.toLowerCase();

                        setLoadProgress({
                            stage: fileName.endsWith(".json") ? "tokenizer" : "model",
                            loaded: 0,
                            total: file.size,
                            fileName: file.name,
                        });

                        if (fileName.endsWith(".json")) {
                            console.log("[Kandle] Reading tokenizer file:", file.name);
                            tokenizerBuffer = await file.arrayBuffer();
                        } else if (fileName.endsWith(".safetensors")) {
                            console.log("[Kandle] Reading model file:", file.name);
                            modelBuffer = await file.arrayBuffer();
                        }
                    }

                    if (!tokenizerBuffer) {
                        throw new Error("No tokenizer.json file selected");
                    }
                    if (!modelBuffer) {
                        throw new Error("No model.safetensors file selected");
                    }

                    console.log(
                        `[Kandle] Files read: tokenizer=${tokenizerBuffer.byteLength}, model=${modelBuffer.byteLength}`
                    );

                    // 通过 Worker 加载（使用 Buffer 模式）
                    await workerManager.loadModelFromBuffer(
                        tokenizerBuffer,
                        modelBuffer,
                        (progress) => {
                            setLoadProgress(progress);
                        }
                    );
                }

                setModelLoaded(true);
                console.log("[Kandle] Model loaded successfully via Worker!");
            } catch (e) {
                // 用户取消文件选择不算错误
                if ((e as Error).name === "AbortError") {
                    console.log("[Kandle] File selection cancelled by user");
                } else {
                    const msg = (e as Error).message;
                    setLoadError(msg);
                    console.error("[Kandle] Model load failed:", msg);
                }
            } finally {
                setIsLoadingModel(false);
                setLoadProgress(null);
            }
        },
        [workerManager]
    );

    // Show loading/error state before main UI
    if (isInitializing) {
        return (
            <div className="min-h-screen bg-grid flex items-center justify-center">
                <div className="text-center">
                    <div className="w-16 h-16 border-4 border-cyber-cyan border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                    <p className="text-gray-400">Initializing WebGPU Worker...</p>
                </div>
            </div>
        );
    }

    if (backendError) {
        return (
            <div className="min-h-screen bg-grid flex items-center justify-center">
                <div className="panel max-w-md mx-auto border-red-500/50">
                    <div className="panel-header text-red-400">WebGPU Error</div>
                    <div className="p-4">
                        <p className="text-red-300 mb-4">{backendError}</p>
                        <p className="text-gray-500 text-sm">
                            This demo requires a WebGPU-enabled browser (Chrome 113+, Edge 113+, or
                            Firefox Nightly with WebGPU flag enabled).
                        </p>
                    </div>
                </div>
            </div>
        );
    }

    // Main UI
    return (
        <InterpretabilityLayout
            useMock={!modelLoaded}
            webgpuAvailable={webgpuInfo?.available ?? false}
            modelLoaded={modelLoaded}
            isLoadingModel={isLoadingModel}
            loadProgress={loadProgress}
            loadError={loadError}
            modelName="Qwen3-0.6B"
            onLoadModel={handleLoadModel}
            modelUrls={QWEN3_MODEL_URLS["0.6b"]}
            workerManager={workerManager}
        />
    );
}

export default App;
