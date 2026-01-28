/**
 * Inference Worker
 *
 * 在 Web Worker 中运行 WebGPU 推理，完全不阻塞主线程
 */

/// <reference lib="webworker" />

declare const self: DedicatedWorkerGlobalScope;

// Worker 中的 navigator 有 gpu 属性
declare const navigator: WorkerNavigator & { gpu?: GPU };

import { env } from "@kandle/core";
import { WebGPUBackend } from "@kandle/backend-webgpu";
import type {
    WorkerCommand,
    WorkerEvent,
    LoadModelPayload,
    LoadModelFromBufferPayload,
    StartGenerationPayload,
    StepPayload,
    LoadProgress,
} from "./message-types";
import { ModelHandler } from "./model-handler";

// ============================================================================
// Worker 全局状态
// ============================================================================

let backend: WebGPUBackend | null = null;
let modelHandler: ModelHandler | null = null;

// ============================================================================
// 消息发送工具
// ============================================================================

function post(event: WorkerEvent): void {
    self.postMessage(event);
}

function postError(operation: string, error: unknown): void {
    const err = error as Error;
    console.error(`[InferenceWorker] Error in ${operation}:`, err);
    post({
        type: "error",
        payload: {
            operation,
            message: err.message || String(error),
            stack: err.stack,
        },
    });
}

// ============================================================================
// 命令处理器
// ============================================================================

async function handleInit(): Promise<void> {
    console.log("[InferenceWorker] Initializing WebGPU...");

    if (!navigator.gpu) {
        throw new Error(
            "WebGPU is not supported in this Worker environment. navigator.gpu is undefined."
        );
    }

    // 创建 WebGPU 后端
    backend = (await WebGPUBackend.create()) as WebGPUBackend;
    env.setBackend(backend);
    env.setDefaultDevice(backend.name);

    console.log("[InferenceWorker] WebGPU initialized successfully");

    post({
        type: "ready",
        payload: {
            available: true,
            supportsF16: false, // TODO: 从 backend 获取
            adapterInfo: "WebGPU",
        },
    });
}

async function handleLoadModel(payload: LoadModelPayload): Promise<void> {
    console.log("[InferenceWorker] Loading model...", payload);

    if (!backend) {
        throw new Error("WebGPU backend not initialized. Call init first.");
    }

    // 创建模型处理器
    modelHandler = new ModelHandler();

    // 加载模型，发送进度事件
    await modelHandler.loadModel(
        payload.tokenizerUrl,
        payload.modelUrl,
        (progress: LoadProgress) => {
            post({ type: "loadProgress", payload: progress });
        }
    );

    console.log("[InferenceWorker] Model loaded successfully");
    post({ type: "modelReady" });
}

async function handleLoadModelFromBuffer(payload: LoadModelFromBufferPayload): Promise<void> {
    console.log("[InferenceWorker] Loading model from buffer...");

    if (!backend) {
        throw new Error("WebGPU backend not initialized. Call init first.");
    }

    // 创建模型处理器
    modelHandler = new ModelHandler();

    // 从 buffer 加载模型，发送进度事件
    await modelHandler.loadModelFromBuffer(payload, (progress: LoadProgress) => {
        post({ type: "loadProgress", payload: progress });
    });

    console.log("[InferenceWorker] Model loaded successfully from buffer");
    post({ type: "modelReady" });
}

async function handleStartGeneration(payload: StartGenerationPayload): Promise<void> {
    console.log("[InferenceWorker] Starting generation...", payload);

    if (!modelHandler) {
        throw new Error("Model not loaded. Call loadModel first.");
    }

    post({ type: "generationStarted" });

    // 初始化生成器并获取第一步预测
    const firstStep = await modelHandler.startGeneration(payload);

    // 发送第一步预测结果，等待主线程调用 step
    post({ type: "generationStep", payload: firstStep });
}

async function handleStep(payload?: StepPayload): Promise<void> {
    if (!modelHandler) {
        throw new Error("Model not loaded. Call loadModel first.");
    }

    const step = await modelHandler.step(payload?.overrideTokenId);

    if (step) {
        post({ type: "generationStep", payload: step });

        if (step.isEos) {
            post({ type: "generationComplete" });
        }
    } else {
        post({ type: "generationComplete" });
    }
}

async function handleUndo(): Promise<void> {
    if (!modelHandler) {
        throw new Error("Model not loaded. Call loadModel first.");
    }

    const step = await modelHandler.undo();

    if (step) {
        post({ type: "undoComplete", payload: step });
    } else {
        post({ type: "undoFailed", payload: { reason: "Cannot undo further" } });
    }
}

function handleStop(): void {
    console.log("[InferenceWorker] Stop requested");
    if (modelHandler) {
        modelHandler.stop();
    }
}

function handleDispose(): void {
    console.log("[InferenceWorker] Disposing resources...");
    if (modelHandler) {
        modelHandler.dispose();
        modelHandler = null;
    }
    backend = null;
}

// ============================================================================
// 消息路由
// ============================================================================

self.onmessage = async (e: MessageEvent<WorkerCommand>) => {
    const { type } = e.data;

    try {
        switch (type) {
            case "init":
                await handleInit();
                break;
            case "loadModel":
                await handleLoadModel(e.data.payload);
                break;
            case "loadModelFromBuffer":
                await handleLoadModelFromBuffer(e.data.payload);
                break;
            case "startGeneration":
                await handleStartGeneration(e.data.payload);
                break;
            case "step":
                await handleStep(e.data.payload);
                break;
            case "undo":
                await handleUndo();
                break;
            case "stop":
                handleStop();
                break;
            case "dispose":
                handleDispose();
                break;
            default:
                console.warn("[InferenceWorker] Unknown command:", type);
        }
    } catch (error) {
        postError(type, error);
    }
};

// Worker 准备就绪的日志
console.log("[InferenceWorker] Worker script loaded");
