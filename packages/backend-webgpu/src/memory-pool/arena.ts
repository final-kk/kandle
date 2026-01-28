/**
 * Arena
 * 
 * 一个 Arena 代表一块大型 GPUBuffer，内部通过偏移进行子分配。
 * 
 * 设计要点：
 * 1. 每个 Arena 有固定的 usage 类型
 * 2. 使用 Bump Allocator 管理内部空间
 * 3. 支持 reset() 用于复用
 */

// ============================================================================
// Types
// ============================================================================

export interface ArenaInfo {
    /** Arena ID */
    id: number;
    /** Buffer Usage Flags */
    usage: GPUBufferUsageFlags;
    /** 总大小（字节） */
    totalSize: number;
    /** 已分配大小（字节） */
    allocatedSize: number;
    /** 活跃 block 数量 */
    activeBlockCount: number;
    /** 已释放但未回收的大小 */
    fragmentedSize: number;
}

interface ActiveBlock {
    allocId: number;
    offset: number;
    size: number;
}

// ============================================================================
// Arena
// ============================================================================

let nextArenaId = 0;

export class Arena {
    readonly id: number;
    readonly usage: GPUBufferUsageFlags;
    readonly totalSize: number;
    readonly buffer: GPUBuffer;

    /** Bump 指针：下一次分配的起始位置 */
    private bumpPointer: number = 0;

    /** 活跃 block 集合 */
    private activeBlocks: Map<number, ActiveBlock> = new Map();

    /** 已释放的总字节数（用于碎片率计算） */
    private freedBytes: number = 0;

    constructor(device: GPUDevice, size: number, usage: GPUBufferUsageFlags) {
        this.id = nextArenaId++;
        this.usage = usage;
        this.totalSize = size;

        this.buffer = device.createBuffer({
            size,
            usage,
            mappedAtCreation: false,
        });
    }

    // ========================================================================
    // Allocation
    // ========================================================================

    /**
     * 分配一块内存
     * 
     * 使用 Bump Allocator，时间复杂度 O(1)
     * 
     * @param size 所需大小（已对齐）
     * @returns 分配成功返回 offset，失败返回 -1
     */
    allocate(size: number, allocId: number): number {
        // 对齐到 256 字节（WebGPU 对齐要求）
        const alignedSize = Math.ceil(size / 256) * 256;

        // 检查空间是否足够
        if (this.bumpPointer + alignedSize > this.totalSize) {
            return -1;  // 空间不足
        }

        const offset = this.bumpPointer;
        this.bumpPointer += alignedSize;

        // 记录活跃 block
        this.activeBlocks.set(allocId, {
            allocId,
            offset,
            size: alignedSize,
        });

        return offset;
    }

    /**
     * 释放一块内存
     * 
     * 注意：这只是标记释放，空间不会立即复用。
     * 需要通过 SegregatedFreeLists 管理复用。
     */
    free(allocId: number): boolean {
        const block = this.activeBlocks.get(allocId);
        if (!block) {
            return false;
        }

        this.activeBlocks.delete(allocId);
        this.freedBytes += block.size;

        return true;
    }

    /**
     * 检查是否有足够空间
     */
    hasSpace(size: number): boolean {
        const alignedSize = Math.ceil(size / 256) * 256;
        return this.bumpPointer + alignedSize <= this.totalSize;
    }

    // ========================================================================
    // Reset & Destroy
    // ========================================================================

    /**
     * 重置 Arena（用于复用）
     * 
     * 不销毁 GPUBuffer，只清空内部状态
     */
    reset(): void {
        this.bumpPointer = 0;
        this.activeBlocks.clear();
        this.freedBytes = 0;
    }

    /**
     * 销毁 Arena 及其 GPUBuffer
     */
    destroy(): void {
        this.buffer.destroy();
        this.activeBlocks.clear();
    }

    // ========================================================================
    // Inspection
    // ========================================================================

    /**
     * 获取所有活跃 block
     */
    getActiveBlocks(): ActiveBlock[] {
        return Array.from(this.activeBlocks.values());
    }

    /**
     * 获取活跃 block 数量
     */
    get activeBlockCount(): number {
        return this.activeBlocks.size;
    }

    /**
     * 获取已分配字节数
     */
    get allocatedBytes(): number {
        return this.bumpPointer;
    }

    /**
     * 获取空闲字节数（未分配的尾部空间）
     */
    get freeBytes(): number {
        return this.totalSize - this.bumpPointer;
    }

    /**
     * 获取已释放字节数（碎片）
     */
    get fragmentedBytes(): number {
        return this.freedBytes;
    }

    /**
     * 获取碎片率
     */
    get fragmentationRatio(): number {
        if (this.bumpPointer === 0) return 0;
        return this.freedBytes / this.bumpPointer;
    }

    /**
     * 检查 Arena 是否为空（没有活跃 block）
     */
    get isEmpty(): boolean {
        return this.activeBlocks.size === 0;
    }

    /**
     * 获取 Arena 信息
     */
    getInfo(): ArenaInfo {
        return {
            id: this.id,
            usage: this.usage,
            totalSize: this.totalSize,
            allocatedSize: this.bumpPointer,
            activeBlockCount: this.activeBlocks.size,
            fragmentedSize: this.freedBytes,
        };
    }
}

/**
 * 重置 Arena ID 计数器（仅用于测试）
 */
export function resetArenaIdCounter(): void {
    nextArenaId = 0;
}
