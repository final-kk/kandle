/**
 * ArenaManager
 * 
 * GPU 内存池的核心管理器。
 * 
 * 职责：
 * 1. 按 Usage 分池管理 Arena
 * 2. 使用 SegregatedFreeLists 实现 O(1) 分配
 * 3. EmptyArenaPool 复用空 Arena
 * 4. OOM 恢复分级策略
 */

import { Arena } from './arena';
import { getAllocationStore, PagedAllocationStore } from './allocation-store';
import { SegregatedFreeLists, getSizeClassIndex, getSizeClassSize } from './free-lists';
import { getLeakDetector } from './leak-detector';
import {
    BufferUsagePreset,
    DEFAULT_ARENA_CONFIG,
    ArenaConfig,
    OutOfMemoryError,
    DEBUG_MODE,
} from './types';

// ============================================================================
// Types
// ============================================================================

export interface AllocationResult {
    /** Allocation ID */
    allocId: number;
    /** Debug Token (DEBUG_MODE only) */
    debugToken: number;
    /** Arena ID */
    arenaId: number;
    /** Offset in Arena */
    offset: number;
    /** Allocated size */
    size: number;
}

export interface ArenaManagerStats {
    /** 按 usage 分组的 Arena 统计 */
    arenasByUsage: Map<GPUBufferUsageFlags, {
        arenaCount: number;
        totalBytes: number;
        usedBytes: number;
        freeBytes: number;
    }>;
    /** 空闲 Arena 池统计 */
    emptyArenaPool: {
        arenaCount: number;
        totalBytes: number;
    };
    /** 总分配统计 */
    allocations: {
        activeCount: number;
        peakCount: number;
    };
}

// ============================================================================
// ArenaManager
// ============================================================================

export class ArenaManager {
    readonly device: GPUDevice;
    private deviceLimits: {
        maxBufferSize: number;
        maxStorageBufferBindingSize: number;
    };

    /** 按 Usage 分组的 Arena 列表 */
    private arenasByUsage: Map<GPUBufferUsageFlags, Arena[]> = new Map();

    /** 按 Usage 分组的 Free Lists */
    private freeListsByUsage: Map<GPUBufferUsageFlags, SegregatedFreeLists> = new Map();

    /** 空 Arena 复用池：key = `${usage}:${size}` */
    private emptyArenaPool: Map<string, Arena[]> = new Map();

    /** 闲置 Arena 的总大小限制 */
    private maxIdleArenaBytes: number = 1024 * 1024 * 1024;  // 1GB
    private currentIdleArenaBytes: number = 0;

    /** Arena 大小配置 */
    private arenaConfig: ArenaConfig;

    /** 分配暂停标志（Compaction 期间使用） */
    private allocationsPaused: boolean = false;

    /** AllocationStore 引用 */
    private allocStore: PagedAllocationStore;

    constructor(device: GPUDevice) {
        this.device = device;

        // 读取设备限制
        this.deviceLimits = {
            maxBufferSize: device.limits.maxBufferSize,
            maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
        };

        // 动态调整 Arena 大小配置
        this.arenaConfig = {
            small: Math.min(DEFAULT_ARENA_CONFIG.small, this.deviceLimits.maxBufferSize),
            medium: Math.min(DEFAULT_ARENA_CONFIG.medium, this.deviceLimits.maxBufferSize),
            large: Math.min(DEFAULT_ARENA_CONFIG.large, this.deviceLimits.maxBufferSize),
        };

        this.allocStore = getAllocationStore();
    }

    // ========================================================================
    // Core Allocation
    // ========================================================================

    /**
     * 分配内存
     * 
     * 分配策略（分级 OOM 恢复）：
     * 1. 从 Free List 取
     * 2. 从现有 Arena 分配
     * 3. 从空 Arena 池复用
     * 4. 创建新 Arena
     */
    allocate(
        size: number,
        usage: GPUBufferUsageFlags = BufferUsagePreset.STORAGE
    ): AllocationResult {
        if (this.allocationsPaused) {
            throw new Error('Allocations are paused (compaction in progress)');
        }

        // 检查 Binding 限制
        if (size > this.deviceLimits.maxStorageBufferBindingSize) {
            throw new OutOfMemoryError(
                `Tensor size ${size} exceeds maxStorageBufferBindingSize ` +
                `(${this.deviceLimits.maxStorageBufferBindingSize}). ` +
                `Consider using tensor tiling or a device with higher limits.`
            );
        }

        // 对齐到 256 字节
        const alignedSize = Math.ceil(size / 256) * 256;

        // Level 1: 从 Free List 取
        const freeLists = this.getOrCreateFreeLists(usage);
        const freeResult = freeLists.popForSize(alignedSize);

        if (freeResult) {
            const { block, needSplit } = freeResult;

            if (needSplit && block.size > alignedSize) {
                // 分割多余的部分放回 Free List
                const remainderOffset = block.offset + alignedSize;
                const remainderSize = block.size - alignedSize;

                // 为 remainder 创建新的 allocId
                const remainderAllocId = this.allocStore.allocId();
                this.allocStore.setArenaId(remainderAllocId, block.arenaId);
                this.allocStore.setOffset(remainderAllocId, remainderOffset);
                this.allocStore.setSize(remainderAllocId, remainderSize);
                // 立即放回 Free List
                this.allocStore.freeId(remainderAllocId);
                freeLists.push(remainderAllocId, block.arenaId, remainderOffset, remainderSize);
            }

            // 复用 free block 的 allocId
            const allocId = block.allocId;
            const debugToken = DEBUG_MODE ? this.allocStore.allocId() : 0;

            // 但我们需要重新分配一个新的 allocId（因为旧的已经在 free 状态）
            const newAllocId = this.allocStore.allocId();
            this.allocStore.setArenaId(newAllocId, block.arenaId);
            this.allocStore.setOffset(newAllocId, block.offset);
            this.allocStore.setSize(newAllocId, alignedSize);

            // Debug: 追踪分配
            if (DEBUG_MODE) {
                getLeakDetector().track(newAllocId, alignedSize);
            }

            return {
                allocId: newAllocId,
                debugToken: this.getDebugToken(newAllocId),
                arenaId: block.arenaId,
                offset: block.offset,
                size: alignedSize,
            };
        }

        // Level 2: 从现有 Arena 分配
        const arenas = this.arenasByUsage.get(usage);
        if (arenas) {
            for (const arena of arenas) {
                if (arena.hasSpace(alignedSize)) {
                    return this.allocateFromArena(arena, alignedSize);
                }
            }
        }

        // Level 3: 从空 Arena 池复用
        const arenaSize = this.chooseArenaSize(alignedSize);
        const emptyArena = this.tryGetFromEmptyPool(usage, arenaSize);
        if (emptyArena) {
            this.addArena(usage, emptyArena);
            return this.allocateFromArena(emptyArena, alignedSize);
        }

        // Level 4: 创建新 Arena
        const newArena = this.createArena(usage, arenaSize);
        this.addArena(usage, newArena);
        return this.allocateFromArena(newArena, alignedSize);
    }

    private allocateFromArena(arena: Arena, size: number): AllocationResult {
        const [allocId, debugToken] = this.allocStore.allocIdWithToken();
        const offset = arena.allocate(size, allocId);

        if (offset === -1) {
            throw new Error('Arena allocation failed unexpectedly');
        }

        this.allocStore.setArenaId(allocId, arena.id);
        this.allocStore.setOffset(allocId, offset);
        this.allocStore.setSize(allocId, size);

        // Debug: 追踪分配
        if (DEBUG_MODE) {
            getLeakDetector().track(allocId, size);
        }

        return {
            allocId,
            debugToken,
            arenaId: arena.id,
            offset,
            size,
        };
    }

    // ========================================================================
    // Deallocation
    // ========================================================================

    /**
     * 释放内存（延迟释放，放入 Free List）
     */
    free(allocId: number): void {
        const arenaId = this.allocStore.getArenaId(allocId);
        const offset = this.allocStore.getOffset(allocId);
        const size = this.allocStore.getSize(allocId);

        if (arenaId === -1) {
            console.warn(`Allocation ${allocId} already freed`);
            return;
        }

        // 找到对应的 Arena
        const arena = this.findArenaById(arenaId);
        if (!arena) {
            console.error(`Arena ${arenaId} not found for allocation ${allocId}`);
            return;
        }

        // 标记 Arena 中的 block 为已释放
        arena.free(allocId);

        // 放入 Free List
        const freeLists = this.getOrCreateFreeLists(arena.usage);
        freeLists.push(allocId, arenaId, offset, size);

        // 标记 allocStore 中的记录为已释放
        this.allocStore.freeId(allocId);

        // Debug: 取消追踪
        if (DEBUG_MODE) {
            getLeakDetector().untrack(allocId);
        }

        // 暂时禁用 Arena 回收，诊断 use-after-free 问题
        // TODO: 需要更安全的 Arena 回收机制，确保没有活跃引用才能回收
        // if (arena.isEmpty) {
        //     this.considerRecyclingArena(arena);
        // }
    }

    // ========================================================================
    // Arena Management
    // ========================================================================

    private getOrCreateFreeLists(usage: GPUBufferUsageFlags): SegregatedFreeLists {
        let freeLists = this.freeListsByUsage.get(usage);
        if (!freeLists) {
            freeLists = new SegregatedFreeLists();
            this.freeListsByUsage.set(usage, freeLists);
        }
        return freeLists;
    }

    private addArena(usage: GPUBufferUsageFlags, arena: Arena): void {
        let arenas = this.arenasByUsage.get(usage);
        if (!arenas) {
            arenas = [];
            this.arenasByUsage.set(usage, arenas);
        }
        arenas.push(arena);
    }

    private findArenaById(arenaId: number): Arena | null {
        for (const arenas of this.arenasByUsage.values()) {
            for (const arena of arenas) {
                if (arena.id === arenaId) {
                    return arena;
                }
            }
        }
        return null;
    }

    private createArena(usage: GPUBufferUsageFlags, size: number): Arena {
        return new Arena(this.device, size, usage);
    }

    private chooseArenaSize(minSize: number): number {
        // 选择能容纳 minSize 的最小配置
        if (minSize <= this.arenaConfig.small / 4) {
            return this.arenaConfig.small;
        } else if (minSize <= this.arenaConfig.medium / 4) {
            return this.arenaConfig.medium;
        } else {
            // 对于大分配，创建刚好足够的 Arena
            return Math.min(
                Math.max(minSize * 2, this.arenaConfig.medium),
                this.arenaConfig.large
            );
        }
    }

    // ========================================================================
    // Empty Arena Pool
    // ========================================================================

    private tryGetFromEmptyPool(usage: GPUBufferUsageFlags, size: number): Arena | null {
        const key = `${usage}:${size}`;
        const pool = this.emptyArenaPool.get(key);

        if (pool && pool.length > 0) {
            const arena = pool.pop()!;
            this.currentIdleArenaBytes -= arena.totalSize;
            return arena;
        }

        return null;
    }

    /**
     * 考虑回收空 Arena
     */
    private considerRecyclingArena(arena: Arena): void {
        // 从活跃列表移除
        const arenas = this.arenasByUsage.get(arena.usage);
        if (arenas) {
            const index = arenas.indexOf(arena);
            if (index !== -1) {
                arenas.splice(index, 1);
            }
        }

        // 从 Free Lists 移除该 Arena 的所有 block
        const freeLists = this.freeListsByUsage.get(arena.usage);
        if (freeLists) {
            freeLists.removeByArena(arena.id);
        }

        // 尝试回收到复用池
        this.recycleEmptyArena(arena);
    }

    /**
     * 回收空 Arena 到复用池
     */
    recycleEmptyArena(arena: Arena): void {
        const key = `${arena.usage}:${arena.totalSize}`;

        // 检查是否超过闲置上限
        if (this.currentIdleArenaBytes + arena.totalSize > this.maxIdleArenaBytes) {
            // 超过上限，真正销毁
            arena.destroy();
            return;
        }

        // 放入复用池
        let pool = this.emptyArenaPool.get(key);
        if (!pool) {
            pool = [];
            this.emptyArenaPool.set(key, pool);
        }

        arena.reset();  // 清空内部分配状态
        pool.push(arena);
        this.currentIdleArenaBytes += arena.totalSize;
    }

    /**
     * 释放所有闲置 Arena（用于 OOM 恢复）
     */
    shrinkIdleArenas(): void {
        for (const [, pool] of this.emptyArenaPool) {
            for (const arena of pool) {
                arena.destroy();
            }
        }
        this.emptyArenaPool.clear();
        this.currentIdleArenaBytes = 0;
    }

    // ========================================================================
    // Compaction Support
    // ========================================================================

    /**
     * 暂停分配（Compaction 期间使用）
     */
    pauseAllocations(): void {
        this.allocationsPaused = true;
    }

    /**
     * 恢复分配
     */
    resumeAllocations(): void {
        this.allocationsPaused = false;
    }

    /**
     * 检查分配是否被暂停
     */
    get isPaused(): boolean {
        return this.allocationsPaused;
    }

    // ========================================================================
    // Buffer Access
    // ========================================================================

    /**
     * 获取分配对应的 GPUBuffer
     */
    getBuffer(allocId: number): GPUBuffer {
        const arenaId = this.allocStore.getArenaId(allocId);
        const arena = this.findArenaById(arenaId);
        if (!arena) {
            throw new Error(`Arena ${arenaId} not found for allocation ${allocId}`);
        }
        return arena.buffer;
    }

    /**
     * 获取分配的偏移量
     */
    getOffset(allocId: number): number {
        return this.allocStore.getOffset(allocId);
    }

    /**
     * 获取分配的大小
     */
    getSize(allocId: number): number {
        return this.allocStore.getSize(allocId);
    }

    // ========================================================================
    // Validation
    // ========================================================================

    /**
     * 校验 allocation ID
     */
    validateAllocation(allocId: number, token: number): boolean {
        return this.allocStore.validateId(allocId, token);
    }

    private getDebugToken(allocId: number): number {
        if (!DEBUG_MODE) return 0;
        // Token 存储在 AllocationStore 内部
        // 这里需要一个方法来获取它
        // 简化起见，我们使用 generation 作为 token 的一部分
        return this.allocStore.getGeneration(allocId);
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    getStats(): ArenaManagerStats {
        const arenasByUsage = new Map<GPUBufferUsageFlags, {
            arenaCount: number;
            totalBytes: number;
            usedBytes: number;
            freeBytes: number;
        }>();

        for (const [usage, arenas] of this.arenasByUsage) {
            let totalBytes = 0;
            let usedBytes = 0;

            for (const arena of arenas) {
                totalBytes += arena.totalSize;
                usedBytes += arena.allocatedBytes;
            }

            arenasByUsage.set(usage, {
                arenaCount: arenas.length,
                totalBytes,
                usedBytes,
                freeBytes: totalBytes - usedBytes,
            });
        }

        let emptyArenaCount = 0;
        let emptyArenaBytes = 0;
        for (const pool of this.emptyArenaPool.values()) {
            emptyArenaCount += pool.length;
            for (const arena of pool) {
                emptyArenaBytes += arena.totalSize;
            }
        }

        return {
            arenasByUsage,
            emptyArenaPool: {
                arenaCount: emptyArenaCount,
                totalBytes: emptyArenaBytes,
            },
            allocations: {
                activeCount: this.allocStore.activeCount,
                peakCount: this.allocStore.peakCount,
            },
        };
    }

    /**
     * 获取设备限制
     */
    getDeviceLimits(): typeof this.deviceLimits {
        return { ...this.deviceLimits };
    }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let instance: ArenaManager | null = null;

export function getArenaManager(): ArenaManager {
    if (!instance) {
        throw new Error('ArenaManager not initialized. Call initializeArenaManager(device) first.');
    }
    return instance;
}

export function initializeArenaManager(device: GPUDevice): ArenaManager {
    if (instance) {
        console.warn('ArenaManager already initialized');
        return instance;
    }
    instance = new ArenaManager(device);
    return instance;
}

/**
 * 重置单例（仅用于测试）
 */
export function resetArenaManager(): void {
    instance = null;
}
