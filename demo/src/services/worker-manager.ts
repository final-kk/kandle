/**
 * Inference Worker Manager
 *
 * 主线程中管理 Inference Worker 的代理类
 * 提供 Promise 风格的 API，隐藏 postMessage 通信细节
 */

import type {
    WorkerCommand,
    WorkerEvent,
    WebGPUInfo,
    LoadProgress,
    GenerationStep,
    StartGenerationPayload,
    ErrorPayload,
} from "../workers/message-types";

// ============================================================================
// Types
// ============================================================================

type EventCallback<T = unknown> = (data: T) => void;

export interface InferenceWorkerManagerEvents {
    ready: WebGPUInfo;
    loadProgress: LoadProgress;
    modelReady: void;
    generationStarted: void;
    generationStep: GenerationStep;
    generationComplete: void;
    undoComplete: GenerationStep;
    undoFailed: { reason: string };
    error: ErrorPayload;
}

// ============================================================================
// InferenceWorkerManager
// ============================================================================

export class InferenceWorkerManager {
    private worker: Worker;
    private eventListeners: Map<string, Set<EventCallback>> = new Map();
    private isInitialized = false;
    private isInitializing = false;
    private initPromise: Promise<WebGPUInfo> | null = null;
    private isModelLoaded = false;

    constructor() {
        // 创建 Worker（Vite 支持的语法）
        this.worker = new Worker(new URL("../workers/inference.worker.ts", import.meta.url), {
            type: "module",
        });

        // 监听 Worker 消息
        this.worker.onmessage = this.handleMessage.bind(this);
        this.worker.onerror = this.handleError.bind(this);

        console.log("[WorkerManager] Worker created");
    }

    // ========================================================================
    // 消息处理
    // ========================================================================

    private handleMessage(e: MessageEvent<WorkerEvent>): void {
        const event = e.data;
        console.log("[WorkerManager] Received event:", event.type);

        // 触发事件监听器
        this.emit(event.type, "payload" in event ? event.payload : undefined);
    }

    private handleError(e: ErrorEvent): void {
        console.error("[WorkerManager] Worker error:", e);
        this.emit("error", {
            operation: "worker",
            message: e.message || "Unknown worker error",
        });
    }

    // ========================================================================
    // 事件系统
    // ========================================================================

    private emit(type: string, data?: unknown): void {
        const listeners = this.eventListeners.get(type);
        if (listeners) {
            listeners.forEach((fn) => fn(data));
        }
    }

    /**
     * 订阅事件
     * @returns 取消订阅的函数
     */
    on<K extends keyof InferenceWorkerManagerEvents>(
        event: K,
        callback: EventCallback<InferenceWorkerManagerEvents[K]>
    ): () => void {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, new Set());
        }
        this.eventListeners.get(event)!.add(callback as EventCallback);

        // 返回取消订阅函数
        return () => {
            this.eventListeners.get(event)?.delete(callback as EventCallback);
        };
    }

    /**
     * 一次性订阅（收到一次后自动取消）
     */
    once<K extends keyof InferenceWorkerManagerEvents>(
        event: K,
        callback: EventCallback<InferenceWorkerManagerEvents[K]>
    ): () => void {
        const off = this.on(event, (data) => {
            off();
            callback(data);
        });
        return off;
    }

    // ========================================================================
    // 命令发送
    // ========================================================================

    private send(command: WorkerCommand): void {
        this.worker.postMessage(command);
    }

    // ========================================================================
    // Public API
    // ========================================================================

    /**
     * 初始化 WebGPU
     */
    async init(): Promise<WebGPUInfo> {
        if (this.isInitialized) {
            console.warn("[WorkerManager] Already initialized");
            return { available: true, supportsF16: false };
        }

        // 防止并发初始化（React Strict Mode 会触发两次 useEffect）
        if (this.isInitializing && this.initPromise) {
            console.warn("[WorkerManager] Already initializing, returning existing promise");
            return this.initPromise;
        }

        this.isInitializing = true;
        this.initPromise = new Promise((resolve, reject) => {
            const cleanup = () => {
                offReady();
                offError();
            };

            const offReady = this.once("ready", (info) => {
                cleanup();
                this.isInitialized = true;
                this.isInitializing = false;
                console.log("[WorkerManager] Initialized:", info);
                resolve(info);
            });

            const offError = this.once("error", (err) => {
                cleanup();
                this.isInitializing = false;
                this.initPromise = null;
                console.error("[WorkerManager] Init error:", err);
                reject(new Error(err.message));
            });

            this.send({ type: "init" });
        });

        return this.initPromise;
    }

    /**
     * 加载模型（URL 模式）
     */
    loadModel(
        tokenizerUrl: string,
        modelUrl: string,
        onProgress?: (progress: LoadProgress) => void
    ): Promise<void> {
        return new Promise((resolve, reject) => {
            const offProgress = onProgress ? this.on("loadProgress", onProgress) : () => {};

            const cleanup = () => {
                offProgress();
                offReady();
                offError();
            };

            const offReady = this.once("modelReady", () => {
                cleanup();
                this.isModelLoaded = true;
                console.log("[WorkerManager] Model loaded");
                resolve();
            });

            const offError = this.once("error", (err) => {
                cleanup();
                console.error("[WorkerManager] Load error:", err);
                reject(new Error(err.message));
            });

            this.send({
                type: "loadModel",
                payload: { tokenizerUrl, modelUrl },
            });
        });
    }

    /**
     * 加载模型（Buffer 模式，用于文件上传）
     */
    loadModelFromBuffer(
        tokenizerBuffer: ArrayBuffer,
        modelBuffer: ArrayBuffer,
        onProgress?: (progress: LoadProgress) => void
    ): Promise<void> {
        return new Promise((resolve, reject) => {
            const offProgress = onProgress ? this.on("loadProgress", onProgress) : () => {};

            const cleanup = () => {
                offProgress();
                offReady();
                offError();
            };

            const offReady = this.once("modelReady", () => {
                cleanup();
                this.isModelLoaded = true;
                console.log("[WorkerManager] Model loaded from buffer");
                resolve();
            });

            const offError = this.once("error", (err) => {
                cleanup();
                console.error("[WorkerManager] Load error:", err);
                reject(new Error(err.message));
            });

            // 使用 Transferable 发送 ArrayBuffer（零拷贝传输）
            this.worker.postMessage(
                {
                    type: "loadModelFromBuffer",
                    payload: { tokenizerBuffer, modelBuffer },
                },
                [tokenizerBuffer, modelBuffer]
            );
        });
    }

    /**
     * 开始生成
     */
    startGeneration(
        config: Omit<StartGenerationPayload, "eosTokenIds"> & { eosTokenIds?: number[] }
    ): void {
        const fullConfig: StartGenerationPayload = {
            ...config,
            eosTokenIds: config.eosTokenIds ?? [151645, 151643],
        };

        this.send({
            type: "startGeneration",
            payload: fullConfig,
        });
    }

    /**
     * 单步生成
     */
    step(overrideTokenId?: number): void {
        this.send({
            type: "step",
            payload: { overrideTokenId },
        });
    }

    /**
     * 后退一步
     *
     * 回滚到上一个状态，重新获取该位置的预测
     */
    undo(): void {
        this.send({ type: "undo" });
    }

    /**
     * 停止生成
     */
    stop(): void {
        this.send({ type: "stop" });
    }

    /**
     * 释放资源并终止 Worker
     */
    dispose(): void {
        this.send({ type: "dispose" });
        this.worker.terminate();
        this.eventListeners.clear();
        this.isInitialized = false;
        this.isModelLoaded = false;
        console.log("[WorkerManager] Disposed");
    }

    // ========================================================================
    // 状态查询
    // ========================================================================

    get initialized(): boolean {
        return this.isInitialized;
    }

    get modelLoaded(): boolean {
        return this.isModelLoaded;
    }
}

// ============================================================================
// 单例管理
// ============================================================================

let instance: InferenceWorkerManager | null = null;

/**
 * 获取 Worker Manager 单例
 */
export function getWorkerManager(): InferenceWorkerManager {
    if (!instance) {
        instance = new InferenceWorkerManager();
    }
    return instance;
}

/**
 * 重置 Worker Manager（用于清理和重新初始化）
 */
export function resetWorkerManager(): void {
    if (instance) {
        instance.dispose();
        instance = null;
    }
}
