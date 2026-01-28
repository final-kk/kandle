/**
 * Uniform Buffer 工具函数
 * 
 * 提供统一的 uniform buffer 创建接口，使用 Pool 管理以避免内存泄漏。
 */

import { UniformBufferPool } from "./UniformBufferPool";
import { WebGPUDeviceManager } from "./device";

/**
 * 创建或复用一个 uniform buffer
 * 
 * 使用 UniformBufferPool 管理，避免每次 kernel 调用都创建新 buffer。
 * 
 * @param data 要写入的数据
 * @returns GPUBuffer
 * 
 * @example
 * ```ts
 * const uniformBuffer = createUniformBuffer(uniformArray);
 * // ... use in bind group ...
 * // Buffer 会在 resetUniformBuffers() 后可复用
 * ```
 */
export function createUniformBuffer(data: ArrayBuffer): GPUBuffer {
    const pool = UniformBufferPool.getInstance();
    return pool.acquire(data.byteLength, data, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
}

/**
 * 重置所有 uniform buffer，标记为可复用
 * 
 * 应在每帧/每次推理迭代结束后调用
 */
export function resetUniformBuffers(): void {
    const pool = UniformBufferPool.getInstance();
    pool.resetFrame();
}

/**
 * 清空所有 uniform buffer
 * 
 * 强制释放所有 GPU 内存
 */
export function clearUniformBuffers(): void {
    const pool = UniformBufferPool.getInstance();
    pool.clear();
}

/**
 * 获取 uniform buffer pool 统计信息
 */
export function getUniformBufferStats() {
    const pool = UniformBufferPool.getInstance();
    return pool.getStats();
}
