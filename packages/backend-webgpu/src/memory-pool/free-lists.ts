/**
 * SegregatedFreeLists
 * 
 * 分离式空闲链表，实现 O(1) 的 block 分配和释放。
 * 
 * 设计要点：
 * 1. 36 级 Size Classes（256B - 2GB）
 * 2. 每级维护独立的空闲链表
 * 3. 支持从更大的 class 借用
 */

import { SIZE_CLASS_CONFIG } from './types';

// ============================================================================
// Size Class Helpers
// ============================================================================

const { NUM_CLASSES, MIN_SIZE, MIN_SIZE_LOG2 } = SIZE_CLASS_CONFIG;

/**
 * 计算给定 size 对应的 size class index
 * 
 * 向上取整到最近的 2 的幂次，然后计算 index
 */
export function getSizeClassIndex(size: number): number {
    if (size <= MIN_SIZE) return 0;

    // 对齐到 2 的幂次
    const log2 = Math.ceil(Math.log2(size));
    const index = log2 - MIN_SIZE_LOG2;

    // Clamp 到有效范围
    return Math.min(Math.max(0, index), NUM_CLASSES - 1);
}

/**
 * 获取 size class 对应的实际大小
 */
export function getSizeClassSize(index: number): number {
    return MIN_SIZE << index;
}

// ============================================================================
// Free Block Entry
// ============================================================================

interface FreeBlock {
    /** Allocation ID (for tracking in AllocationStore) */
    allocId: number;
    /** Arena ID */
    arenaId: number;
    /** Offset in Arena */
    offset: number;
    /** Actual size */
    size: number;
}

// ============================================================================
// SegregatedFreeLists
// ============================================================================

export class SegregatedFreeLists {
    /**
     * 每个 size class 的空闲 block 列表
     * 
     * 使用 Array 而非链表，因为：
     * 1. 数组 push/pop 是 O(1)
     * 2. JS 数组内存连续，缓存友好
     */
    private lists: FreeBlock[][] = [];

    /** 总空闲字节数 */
    private _totalFreeBytes: number = 0;

    /** 总空闲 block 数 */
    private _totalFreeBlocks: number = 0;

    constructor() {
        // 初始化所有 size class 的空列表
        for (let i = 0; i < NUM_CLASSES; i++) {
            this.lists.push([]);
        }
    }

    // ========================================================================
    // Core Operations
    // ========================================================================

    /**
     * 添加 block 到对应的 size class
     * 
     * 时间复杂度: O(1)
     */
    push(allocId: number, arenaId: number, offset: number, size: number): void {
        const classIndex = getSizeClassIndex(size);

        this.lists[classIndex].push({
            allocId,
            arenaId,
            offset,
            size,
        });

        this._totalFreeBytes += size;
        this._totalFreeBlocks++;
    }

    /**
     * 从指定 size class 取一个 block
     * 
     * 时间复杂度: O(1)
     * 
     * @returns FreeBlock 或 null（如果没有）
     */
    pop(sizeClass: number): FreeBlock | null {
        if (sizeClass < 0 || sizeClass >= NUM_CLASSES) {
            return null;
        }

        const list = this.lists[sizeClass];
        if (list.length === 0) {
            return null;
        }

        const block = list.pop()!;
        this._totalFreeBytes -= block.size;
        this._totalFreeBlocks--;

        return block;
    }

    /**
     * 查找并取出能满足 size 的 block
     * 
     * 策略：从 exact fit 开始，向上查找更大的 class
     * 
     * 时间复杂度: O(NUM_CLASSES) 最坏，O(1) 平均
     * 
     * @returns [block, needSplit] 或 null
     *   - needSplit: 如果 block 来自更大的 class，需要分割
     */
    popForSize(size: number): { block: FreeBlock; needSplit: boolean } | null {
        const targetClass = getSizeClassIndex(size);

        // 首先尝试 exact fit
        const exactBlock = this.pop(targetClass);
        if (exactBlock) {
            return { block: exactBlock, needSplit: false };
        }

        // 向上查找更大的 class
        for (let i = targetClass + 1; i < NUM_CLASSES; i++) {
            const largerBlock = this.pop(i);
            if (largerBlock) {
                return { block: largerBlock, needSplit: true };
            }
        }

        return null;
    }

    /**
     * 从指定 Arena 移除所有 block
     * 
     * 用于 Arena 销毁或 Compaction
     * 
     * @returns 被移除的 block 列表
     */
    removeByArena(arenaId: number): FreeBlock[] {
        const removed: FreeBlock[] = [];

        for (let i = 0; i < NUM_CLASSES; i++) {
            const list = this.lists[i];
            const remaining: FreeBlock[] = [];

            for (const block of list) {
                if (block.arenaId === arenaId) {
                    removed.push(block);
                    this._totalFreeBytes -= block.size;
                    this._totalFreeBlocks--;
                } else {
                    remaining.push(block);
                }
            }

            this.lists[i] = remaining;
        }

        return removed;
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    /**
     * 获取总空闲字节数
     */
    get totalFreeBytes(): number {
        return this._totalFreeBytes;
    }

    /**
     * 获取总空闲 block 数
     */
    get totalFreeBlocks(): number {
        return this._totalFreeBlocks;
    }

    /**
     * 检查是否为空
     */
    get isEmpty(): boolean {
        return this._totalFreeBlocks === 0;
    }

    /**
     * 获取各 size class 的 block 数量
     */
    getClassCounts(): number[] {
        return this.lists.map(list => list.length);
    }

    /**
     * 获取详细统计信息
     */
    getStats(): {
        totalFreeBytes: number;
        totalFreeBlocks: number;
        classCounts: { size: number; count: number }[];
    } {
        return {
            totalFreeBytes: this._totalFreeBytes,
            totalFreeBlocks: this._totalFreeBlocks,
            classCounts: this.lists.map((list, i) => ({
                size: getSizeClassSize(i),
                count: list.length,
            })).filter(c => c.count > 0),
        };
    }

    // ========================================================================
    // Clear
    // ========================================================================

    /**
     * 清空所有 free lists
     */
    clear(): void {
        for (let i = 0; i < NUM_CLASSES; i++) {
            this.lists[i] = [];
        }
        this._totalFreeBytes = 0;
        this._totalFreeBlocks = 0;
    }
}
