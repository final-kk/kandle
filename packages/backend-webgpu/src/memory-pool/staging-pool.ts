/**
 * StagingPool
 * 
 * 独立的 Staging Buffer 池，用于 dataAsync() 读回操作。
 * 
 * 设计要点：
 * 1. Staging Buffer 使用 MAP_READ | COPY_DST
 * 2. 按大小分组复用，避免频繁创建
 * 3. 与 Arena 分离，避免锁住计算用 Arena
 */

// ============================================================================
// Constants
// ============================================================================

/** 最小 Staging Buffer 大小 (256 bytes) */
const MIN_STAGING_SIZE = 256;

/** 每个大小桶的最大缓存数量 */
const MAX_CACHED_PER_SIZE = 8;

// ============================================================================
// StagingPool
// ============================================================================

export class StagingPool {
    private device: GPUDevice;

    /** 空闲 Staging Buffer 池（按大小分组） */
    private freeBuffers: Map<number, GPUBuffer[]> = new Map();

    /** 统计信息 */
    private _acquireCount: number = 0;
    private _createCount: number = 0;
    private _reuseCount: number = 0;

    constructor(device: GPUDevice) {
        this.device = device;
    }

    // ========================================================================
    // Size Alignment
    // ========================================================================

    /**
     * 对齐大小到 256 向上取整到 2 的幂次
     */
    private roundSize(size: number): number {
        // 至少 256 bytes
        if (size <= MIN_STAGING_SIZE) return MIN_STAGING_SIZE;

        // 对齐到 256
        const aligned = Math.ceil(size / 256) * 256;

        // 向上取整到 2 的幂次
        return 1 << Math.ceil(Math.log2(aligned));
    }

    // ========================================================================
    // Core Operations
    // ========================================================================

    /**
     * 获取临时 Staging Buffer
     * 
     * @param size 所需大小
     * @returns GPUBuffer，调用者需在使用完后调用 release()
     */
    acquire(size: number): GPUBuffer {
        this._acquireCount++;

        const roundedSize = this.roundSize(size);

        const pool = this.freeBuffers.get(roundedSize);
        if (pool && pool.length > 0) {
            this._reuseCount++;
            return pool.pop()!;
        }

        // 创建新的
        this._createCount++;
        return this.device.createBuffer({
            size: roundedSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
    }

    /**
     * 归还 Staging Buffer
     * 
     * @param buffer 要归还的 Buffer
     */
    release(buffer: GPUBuffer): void {
        const size = buffer.size;

        let pool = this.freeBuffers.get(size);
        if (!pool) {
            pool = [];
            this.freeBuffers.set(size, pool);
        }

        // 限制每个大小桶的缓存数量
        if (pool.length >= MAX_CACHED_PER_SIZE) {
            buffer.destroy();
            return;
        }

        pool.push(buffer);
    }

    // ========================================================================
    // High-Level API
    // ========================================================================

    /**
     * 从 GPU Buffer 读取数据
     * 
     * 这是一个完整的读取流程：
     * 1. 获取 staging buffer
     * 2. 提交 copy 命令
     * 3. map 并读取
     * 4. 归还 staging buffer
     * 
     * @param source 源 Buffer
     * @param sourceOffset 源起始偏移
     * @param size 读取大小
     * @returns ArrayBuffer 包含读取的数据
     */
    async readFromBuffer(
        source: GPUBuffer,
        sourceOffset: number,
        size: number
    ): Promise<ArrayBuffer> {
        // 对齐到 4 字节（WebGPU 要求）
        const alignedSize = Math.ceil(size / 4) * 4;

        const stagingBuffer = this.acquire(alignedSize);

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(
            source, sourceOffset,
            stagingBuffer, 0,
            alignedSize
        );
        this.device.queue.submit([encoder.finish()]);

        // Map 读取
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const mappedRange = stagingBuffer.getMappedRange(0, alignedSize);

        // 复制数据（getMappedRange 返回的是绑定到 GPUBuffer 的视图）
        const data = mappedRange.slice(0, size);

        stagingBuffer.unmap();

        // 归还 staging buffer
        this.release(stagingBuffer);

        return data;
    }

    // ========================================================================
    // Cleanup
    // ========================================================================

    /**
     * 清空所有缓存的 Buffer
     */
    clear(): void {
        for (const [, pool] of this.freeBuffers) {
            for (const buffer of pool) {
                buffer.destroy();
            }
        }
        this.freeBuffers.clear();
    }

    /**
     * 收缩缓存（保留每个大小桶最多 N 个）
     */
    shrink(maxPerSize: number = 2): void {
        for (const [size, pool] of this.freeBuffers) {
            while (pool.length > maxPerSize) {
                const buffer = pool.pop()!;
                buffer.destroy();
            }
            if (pool.length === 0) {
                this.freeBuffers.delete(size);
            }
        }
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    getStats(): {
        buckets: number;
        cachedBuffers: number;
        cachedBytes: number;
        acquireCount: number;
        createCount: number;
        reuseCount: number;
        hitRate: number;
    } {
        let cachedBuffers = 0;
        let cachedBytes = 0;

        for (const [size, pool] of this.freeBuffers) {
            cachedBuffers += pool.length;
            cachedBytes += size * pool.length;
        }

        const hitRate = this._acquireCount > 0
            ? this._reuseCount / this._acquireCount
            : 0;

        return {
            buckets: this.freeBuffers.size,
            cachedBuffers,
            cachedBytes,
            acquireCount: this._acquireCount,
            createCount: this._createCount,
            reuseCount: this._reuseCount,
            hitRate,
        };
    }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let instance: StagingPool | null = null;

export function getStagingPool(): StagingPool {
    if (!instance) {
        throw new Error('StagingPool not initialized. Call initializeStagingPool(device) first.');
    }
    return instance;
}

export function initializeStagingPool(device: GPUDevice): StagingPool {
    if (instance) {
        console.warn('StagingPool already initialized');
        return instance;
    }
    instance = new StagingPool(device);
    return instance;
}

/**
 * 重置单例（仅用于测试）
 */
export function resetStagingPool(): void {
    if (instance) {
        instance.clear();
    }
    instance = null;
}
