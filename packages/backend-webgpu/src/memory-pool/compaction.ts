/**
 * Compaction Engine
 * 
 * Stop-the-world 碎片整理引擎。
 * 
 * 设计要点：
 * 1. Stop-the-world：Compaction 期间禁止新分配
 * 2. 双 await：提交前等一次，提交后等一次
 * 3. 先 copy 后改 offset：永远不会读到垃圾数据
 * 4. Arena 复用：不销毁，放入空 Arena 池
 */

import { Arena } from './arena';
import { getAllocationStore } from './allocation-store';

// ============================================================================
// Types
// ============================================================================

interface MigrationRecord {
    allocId: number;
    newArenaId: number;
    newOffset: number;
}

export interface CompactionResult {
    /** 迁移的 block 数量 */
    migratedBlocks: number;
    /** 迁移的总字节数 */
    migratedBytes: number;
    /** 回收的 Arena 数量 */
    recycledArenas: number;
    /** Compaction 耗时（毫秒） */
    durationMs: number;
}

// ============================================================================
// Compaction Engine
// ============================================================================

export class CompactionEngine {
    private device: GPUDevice;

    constructor(device: GPUDevice) {
        this.device = device;
    }

    // ========================================================================
    // Fragmentation Detection
    // ========================================================================

    /**
     * 检查 Arena 是否需要 Compaction
     * 
     * 条件：
     * 1. 碎片率 > 30%
     * 2. 碎片字节数 > 阈值（如 16MB）
     */
    shouldCompact(arena: Arena): boolean {
        const fragmentationThreshold = 0.3;  // 30% 碎片率
        const minFragmentedBytes = 16 * 1024 * 1024;  // 16MB

        return (
            arena.fragmentationRatio > fragmentationThreshold &&
            arena.fragmentedBytes > minFragmentedBytes
        );
    }

    // ========================================================================
    // Compaction Execution
    // ========================================================================

    /**
     * 执行 Arena Compaction
     * 
     * 将 source Arena 中的所有活跃 block 迁移到 target Arena
     * 
     * @param source 源 Arena（将被清空）
     * @param target 目标 Arena（必须有足够空间）
     * @param pauseFn 暂停分配的回调
     * @param resumeFn 恢复分配的回调
     * @param recycleArenaFn 回收空 Arena 的回调
     */
    async compactArena(
        source: Arena,
        target: Arena,
        pauseFn: () => void,
        resumeFn: () => void,
        recycleArenaFn: (arena: Arena) => void
    ): Promise<CompactionResult> {
        const startTime = performance.now();
        const allocStore = getAllocationStore();

        // ========== Phase 1: Stop-the-world ==========
        pauseFn();

        // 等待所有正在执行的 GPU 任务完成
        await this.device.queue.onSubmittedWorkDone();

        // ========== Phase 2: 收集并拷贝所有活跃 block ==========
        const encoder = this.device.createCommandEncoder();
        const migrations: MigrationRecord[] = [];
        let migratedBytes = 0;

        const activeBlocks = source.getActiveBlocks();

        for (const block of activeBlocks) {
            // 在 target 中分配空间
            const newAllocId = allocStore.allocId();
            const newOffset = target.allocate(block.size, newAllocId);

            if (newOffset === -1) {
                // 目标 Arena 空间不足，这不应该发生
                throw new Error(
                    `Target arena has insufficient space for compaction. ` +
                    `Needed: ${block.size}, Available: ${target.freeBytes}`
                );
            }

            // 提交 copy 命令
            encoder.copyBufferToBuffer(
                source.buffer, block.offset,
                target.buffer, newOffset,
                block.size
            );

            // 记录待更新的映射（先不改！）
            migrations.push({
                allocId: block.allocId,
                newArenaId: target.id,
                newOffset,
            });

            migratedBytes += block.size;

            // 释放临时 allocId
            allocStore.freeId(newAllocId);
        }

        this.device.queue.submit([encoder.finish()]);

        // ========== Phase 3: 等待 Copy 完成 ==========
        await this.device.queue.onSubmittedWorkDone();

        // ========== Phase 4: 原子切换（现在才安全！）==========
        for (const m of migrations) {
            allocStore.setArenaId(m.allocId, m.newArenaId);
            allocStore.setOffset(m.allocId, m.newOffset);
        }

        // ========== Phase 5: 回收旧 Arena ==========
        // 不销毁，放入空 Arena 池复用
        recycleArenaFn(source);

        // ========== Phase 6: 恢复分配 ==========
        resumeFn();

        const durationMs = performance.now() - startTime;

        return {
            migratedBlocks: migrations.length,
            migratedBytes,
            recycledArenas: 1,
            durationMs,
        };
    }

    /**
     * 批量 Compaction：将多个高碎片 Arena 合并到新 Arena
     */
    async compactMultiple(
        sources: Arena[],
        createArenaFn: (size: number) => Arena,
        pauseFn: () => void,
        resumeFn: () => void,
        recycleArenaFn: (arena: Arena) => void
    ): Promise<CompactionResult> {
        if (sources.length === 0) {
            return {
                migratedBlocks: 0,
                migratedBytes: 0,
                recycledArenas: 0,
                durationMs: 0,
            };
        }

        const startTime = performance.now();
        const allocStore = getAllocationStore();

        // ========== Phase 1: Stop-the-world ==========
        pauseFn();
        await this.device.queue.onSubmittedWorkDone();

        // ========== Phase 2: 计算需要的空间并创建目标 Arena ==========
        let totalNeededSpace = 0;
        const allBlocks: { source: Arena; block: { allocId: number; offset: number; size: number } }[] = [];

        for (const source of sources) {
            for (const block of source.getActiveBlocks()) {
                totalNeededSpace += block.size;
                allBlocks.push({ source, block });
            }
        }

        // 创建足够大的目标 Arena
        const target = createArenaFn(totalNeededSpace);

        // ========== Phase 3: 批量拷贝 ==========
        const encoder = this.device.createCommandEncoder();
        const migrations: MigrationRecord[] = [];
        let migratedBytes = 0;

        for (const { source, block } of allBlocks) {
            const newAllocId = allocStore.allocId();
            const newOffset = target.allocate(block.size, newAllocId);

            if (newOffset === -1) {
                throw new Error('Target arena space calculation error');
            }

            encoder.copyBufferToBuffer(
                source.buffer, block.offset,
                target.buffer, newOffset,
                block.size
            );

            migrations.push({
                allocId: block.allocId,
                newArenaId: target.id,
                newOffset,
            });

            migratedBytes += block.size;
            allocStore.freeId(newAllocId);
        }

        this.device.queue.submit([encoder.finish()]);

        // ========== Phase 4: 等待并原子切换 ==========
        await this.device.queue.onSubmittedWorkDone();

        for (const m of migrations) {
            allocStore.setArenaId(m.allocId, m.newArenaId);
            allocStore.setOffset(m.allocId, m.newOffset);
        }

        // ========== Phase 5: 回收旧 Arena ==========
        for (const source of sources) {
            recycleArenaFn(source);
        }

        // ========== Phase 6: 恢复分配 ==========
        resumeFn();

        const durationMs = performance.now() - startTime;

        return {
            migratedBlocks: migrations.length,
            migratedBytes,
            recycledArenas: sources.length,
            durationMs,
        };
    }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let instance: CompactionEngine | null = null;

export function getCompactionEngine(): CompactionEngine {
    if (!instance) {
        throw new Error('CompactionEngine not initialized. Call initializeCompactionEngine(device) first.');
    }
    return instance;
}

export function initializeCompactionEngine(device: GPUDevice): CompactionEngine {
    if (instance) {
        console.warn('CompactionEngine already initialized');
        return instance;
    }
    instance = new CompactionEngine(device);
    return instance;
}

/**
 * 重置单例（仅用于测试）
 */
export function resetCompactionEngine(): void {
    instance = null;
}
