/**
 * Uniform Buffer Pool for WebGPU
 * 
 * 管理 kernel 使用的 uniform buffer，避免每次执行都创建新的 buffer。
 * 
 * 策略：
 * 1. 按大小分桶复用 buffer
 * 2. 使用 FinalizationRegistry 作为安全网
 * 3. 提供 reset() 方法在每帧/每次推理后释放所有 buffer
 */

import { Logger } from "@kandle/utils";

const logger = new Logger('UniformBufferPool');

// Align size to 256 bytes (WebGPU minimum buffer alignment)
function alignSize(size: number): number {
    return Math.ceil(size / 256) * 256;
}

interface PooledUniformBuffer {
    buffer: GPUBuffer;
    size: number;
    inUse: boolean;
}

/**
 * Uniform Buffer Pool 单例
 * 
 * 用于管理 kernel 执行时创建的 uniform buffer，
 * 避免每次 kernel 调用都创建新 buffer 导致的内存泄漏。
 */
export class UniformBufferPool {
    private static instance: UniformBufferPool | null = null;
    private static device: GPUDevice | null = null;

    private pools: Map<number, PooledUniformBuffer[]> = new Map();
    private totalAllocated: number = 0;
    private totalReused: number = 0;

    private constructor() { }

    /**
     * 初始化 Pool
     */
    static initialize(device: GPUDevice): void {
        UniformBufferPool.device = device;
        if (!UniformBufferPool.instance) {
            UniformBufferPool.instance = new UniformBufferPool();
        }
    }

    /**
     * 获取 Pool 实例
     */
    static getInstance(): UniformBufferPool {
        if (!UniformBufferPool.instance) {
            throw new Error('UniformBufferPool not initialized. Call initialize(device) first.');
        }
        return UniformBufferPool.instance;
    }

    /**
     * 获取一个 uniform buffer
     * 
     * @param size 所需大小
     * @param data 要写入的数据（可选）
     * @param usage Buffer 用途（默认 STORAGE | COPY_DST）
     * @returns GPUBuffer
     */
    acquire(size: number, data?: ArrayBuffer, usage?: GPUBufferUsageFlags): GPUBuffer {
        if (!UniformBufferPool.device) {
            throw new Error('UniformBufferPool device not set');
        }

        const bufferUsage = usage ?? (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        const alignedSize = alignSize(size);
        // 使用 size + usage 作为 pool key，确保不同 usage 的 buffer 分开管理
        const poolKey = alignedSize * 1000 + (bufferUsage & 0xFF);
        const pool = this.pools.get(poolKey);

        if (pool) {
            // 查找可用的 buffer
            for (const pooled of pool) {
                if (!pooled.inUse) {
                    pooled.inUse = true;
                    this.totalReused++;

                    // 写入数据
                    if (data) {
                        UniformBufferPool.device.queue.writeBuffer(pooled.buffer, 0, data);
                    }

                    return pooled.buffer;
                }
            }
        }

        // 没有可用的，创建新的
        this.totalAllocated++;
        const buffer = UniformBufferPool.device.createBuffer({
            size: alignedSize,
            usage: bufferUsage,
            mappedAtCreation: data ? true : false,
        });

        if (data) {
            new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data));
            buffer.unmap();
        }

        const pooled: PooledUniformBuffer = {
            buffer,
            size: alignedSize,
            inUse: true,
        };

        if (!this.pools.has(poolKey)) {
            this.pools.set(poolKey, []);
        }
        this.pools.get(poolKey)!.push(pooled);

        return buffer;
    }

    /**
     * 释放一个 uniform buffer (标记为可复用)
     */
    release(buffer: GPUBuffer): void {
        for (const [, pool] of this.pools) {
            for (const pooled of pool) {
                if (pooled.buffer === buffer) {
                    pooled.inUse = false;
                    return;
                }
            }
        }
    }

    /**
     * 标记所有 buffer 为可复用
     * 
     * 在每帧/每次推理迭代结束后调用
     */
    resetFrame(): void {
        for (const [, pool] of this.pools) {
            for (const pooled of pool) {
                pooled.inUse = false;
            }
        }
    }

    /**
     * 清空所有 buffer
     */
    clear(): void {
        for (const [, pool] of this.pools) {
            for (const pooled of pool) {
                pooled.buffer.destroy();
            }
        }
        this.pools.clear();
        logger.debug(`UniformBufferPool cleared. Stats: allocated=${this.totalAllocated}, reused=${this.totalReused}`);
    }

    /**
     * 获取统计信息
     */
    getStats(): { buckets: number; totalBuffers: number; inUse: number; allocated: number; reused: number } {
        let totalBuffers = 0;
        let inUse = 0;

        for (const [, pool] of this.pools) {
            totalBuffers += pool.length;
            inUse += pool.filter(p => p.inUse).length;
        }

        return {
            buckets: this.pools.size,
            totalBuffers,
            inUse,
            allocated: this.totalAllocated,
            reused: this.totalReused,
        };
    }

    /**
     * 重置 Pool（用于设备重置时）
     */
    static reset(): void {
        if (UniformBufferPool.instance) {
            UniformBufferPool.instance.clear();
        }
        UniformBufferPool.instance = null;
    }
}

/**
 * 获取 UniformBufferPool 实例的便捷函数
 */
export function getUniformBufferPool(): UniformBufferPool {
    return UniformBufferPool.getInstance();
}
