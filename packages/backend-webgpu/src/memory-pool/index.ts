/**
 * GPU Memory Pool
 * 
 * 入口模块，提供统一的初始化和访问接口。
 * 
 * 使用方法：
 * 1. 初始化：await initializeMemoryPool(device)
 * 2. 分配：const result = getMemoryPool().allocate(size)
 * 3. 释放：getMemoryPool().release(allocId)
 */

export * from './types';
export * from './allocation-store';
export * from './free-lists';
export * from './arena';
export * from './arena-manager';
export * from './staging-pool';
export * from './fence-tracker';
export * from './compaction';
export * from './leak-detector';
export * from './pooled-storage';

import { getArenaManager, initializeArenaManager, resetArenaManager } from './arena-manager';
import { getStagingPool, initializeStagingPool, resetStagingPool } from './staging-pool';
import {
    getFenceTracker,
    initializeFenceTracker,
    initializeDeferredReleaseQueue,
    resetFenceTracker,
    getDeferredReleaseQueue,
} from './fence-tracker';
import { initializeCompactionEngine, resetCompactionEngine } from './compaction';
import { resetAllocationStore, getAllocationStore } from './allocation-store';
import { resetLeakDetector, getLeakDetector, debug } from './leak-detector';
import { BufferUsagePreset, DEBUG_MODE } from './types';
import { resetArenaIdCounter } from './arena';

// ============================================================================
// Unified Memory Pool Interface
// ============================================================================

export interface MemoryPoolAllocation {
    /** Allocation ID */
    allocId: number;
    /** Debug Token (DEBUG_MODE only) */
    debugToken: number;
    /** GPU Buffer */
    buffer: GPUBuffer;
    /** Offset in buffer */
    offset: number;
    /** Allocated size */
    size: number;
}

/**
 * 统一的内存池接口
 */
export class MemoryPool {
    private device: GPUDevice;
    private initialized: boolean = false;

    constructor(device: GPUDevice) {
        this.device = device;
    }

    /**
     * 初始化所有子系统
     */
    initialize(): void {
        if (this.initialized) {
            console.warn('MemoryPool already initialized');
            return;
        }

        // 初始化各子系统
        initializeArenaManager(this.device);
        initializeStagingPool(this.device);
        initializeFenceTracker(this.device);
        initializeCompactionEngine(this.device);

        // 初始化 DeferredReleaseQueue
        initializeDeferredReleaseQueue(
            getFenceTracker(),
            (allocId) => {
                // 真正释放到 Free List
                getArenaManager().free(allocId);
            }
        );

        this.initialized = true;
    }

    /**
     * 分配内存
     */
    allocate(
        size: number,
        usage: GPUBufferUsageFlags = BufferUsagePreset.STORAGE
    ): MemoryPoolAllocation {
        const arenaManager = getArenaManager();
        const result = arenaManager.allocate(size, usage);

        return {
            allocId: result.allocId,
            debugToken: result.debugToken,
            buffer: arenaManager.getBuffer(result.allocId),
            offset: result.offset,
            size: result.size,
        };
    }

    /**
     * 增加引用计数
     * 
     * 用于 createView 等共享 Storage 的场景
     */
    addRef(allocId: number): void {
        const store = getAllocationStore();
        if (DEBUG_MODE) {
            const refCount = store.getRefCount(allocId);
            if (refCount === 0) {
                throw new Error(`Cannot addRef on freed allocation ${allocId}`);
            }
        }
        store.incrementRefCount(allocId);
    }

    /**
     * 释放引用（refCount--）
     * 
     * 如果 refCount 归零，入队延迟释放等待 GPU 完成
     */
    release(allocId: number): void {
        const store = getAllocationStore();
        const newRefCount = store.decrementRefCount(allocId);

        if (newRefCount === 0) {
            // refCount 归零，入队延迟释放
            const fenceTracker = getFenceTracker();
            const deferredQueue = getDeferredReleaseQueue();
            deferredQueue.enqueue(allocId, fenceTracker.epoch);
        } else if (newRefCount < 0) {
            // Double release 检测
            if (DEBUG_MODE) {
                console.warn(`Double release detected for allocation ${allocId} (refCount=${newRefCount})`);
            }
            // 恢复到 0，避免进一步腐败
            store.setRefCount(allocId, 0);
        }
    }

    /**
     * 立即释放内存（不等待 GPU）
     * 
     * ⚠️ 危险：只有确定 GPU 没有使用此内存时才能调用
     */
    releaseImmediate(allocId: number): void {
        getArenaManager().free(allocId);
    }

    /**
     * 获取分配信息
     */
    getAllocationInfo(allocId: number): {
        buffer: GPUBuffer;
        offset: number;
        size: number;
    } {
        const arenaManager = getArenaManager();
        return {
            buffer: arenaManager.getBuffer(allocId),
            offset: arenaManager.getOffset(allocId),
            size: arenaManager.getSize(allocId),
        };
    }

    /**
     * 校验分配有效性
     */
    validate(allocId: number, token: number): boolean {
        return getArenaManager().validateAllocation(allocId, token);
    }

    /**
     * 从 GPU 读取数据
     */
    async readBuffer(
        allocId: number,
        size?: number
    ): Promise<ArrayBuffer> {
        const arenaManager = getArenaManager();
        const buffer = arenaManager.getBuffer(allocId);
        const offset = arenaManager.getOffset(allocId);
        const allocSize = arenaManager.getSize(allocId);

        return getStagingPool().readFromBuffer(
            buffer,
            offset,
            size ?? allocSize
        );
    }

    /**
     * 写入数据到 GPU
     */
    writeBuffer(allocId: number, data: ArrayBuffer): void {
        const arenaManager = getArenaManager();
        const buffer = arenaManager.getBuffer(allocId);
        const offset = arenaManager.getOffset(allocId);

        this.device.queue.writeBuffer(buffer, offset, data);
    }

    /**
     * 记录 GPU 命令提交
     */
    recordSubmit(): number {
        return getFenceTracker().submit();
    }

    /**
     * 提交命令并记录
     */
    submitCommands(commandBuffers: GPUCommandBuffer[]): number {
        return getFenceTracker().submitCommands(commandBuffers);
    }

    /**
     * Tick：处理延迟释放
     * 
     * 建议每帧调用一次
     */
    tick(): void {
        getDeferredReleaseQueue().tick();

        // Debug 模式下推进帧计数
        if (DEBUG_MODE) {
            getLeakDetector().tick();
        }
    }

    /**
     * 强制同步：等待所有 GPU 命令完成并处理所有延迟释放
     */
    async sync(): Promise<void> {
        await getDeferredReleaseQueue().flushAll();
    }

    /**
     * OOM 恢复：释放所有闲置资源
     */
    shrink(): void {
        getArenaManager().shrinkIdleArenas();
        getStagingPool().shrink();
    }

    /**
     * 获取统计信息
     */
    getStats(): {
        arenaManager: ReturnType<typeof getArenaManager.prototype.getStats>;
        stagingPool: ReturnType<typeof getStagingPool.prototype.getStats>;
        fenceTracker: { epoch: number; confirmed: number; pending: number };
        deferredQueue: { length: number };
    } {
        const fenceTracker = getFenceTracker();

        return {
            arenaManager: getArenaManager().getStats(),
            stagingPool: getStagingPool().getStats(),
            fenceTracker: {
                epoch: fenceTracker.epoch,
                confirmed: fenceTracker.confirmed,
                pending: fenceTracker.pendingCount,
            },
            deferredQueue: {
                length: getDeferredReleaseQueue().length,
            },
        };
    }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let memoryPoolInstance: MemoryPool | null = null;

/**
 * 获取内存池实例
 */
export function getMemoryPool(): MemoryPool {
    if (!memoryPoolInstance) {
        throw new Error('MemoryPool not initialized. Call initializeMemoryPool(device) first.');
    }
    return memoryPoolInstance;
}

/**
 * 初始化内存池
 */
export function initializeMemoryPool(device: GPUDevice): MemoryPool {
    if (memoryPoolInstance) {
        console.warn('MemoryPool already initialized');
        return memoryPoolInstance;
    }

    memoryPoolInstance = new MemoryPool(device);
    memoryPoolInstance.initialize();

    return memoryPoolInstance;
}

/**
 * 检查内存池是否已初始化
 */
export function isMemoryPoolInitialized(): boolean {
    return memoryPoolInstance !== null;
}

/**
 * 重置内存池（仅用于测试）
 */
export function resetMemoryPool(): void {
    resetArenaManager();
    resetStagingPool();
    resetFenceTracker();
    resetCompactionEngine();
    resetAllocationStore();
    resetLeakDetector();
    resetArenaIdCounter();
    memoryPoolInstance = null;
}

// Re-export debug namespace
export { debug };
