/**
 * PagedAllocationStore
 * 
 * 使用分页 TypedArray 存储 Allocation 元数据，避免 JS 对象 GC 开销。
 * 
 * 设计要点：
 * 1. Struct of Arrays (SoA) 布局
 * 2. 分页结构支持动态增长
 * 3. 空闲槽位链表实现 O(1) ID 分配/回收
 * 4. Debug 模式下支持 Token 校验
 */

import { DEBUG_MODE, InvalidAllocationError } from './types';

// ============================================================================
// Constants
// ============================================================================

/** 每页的 slot 数量 (4096) */
const PAGE_SIZE = 4096;

/** 无效 ID 标记 */
const INVALID_ID = -1;

// ============================================================================
// SoA 字段索引
// ============================================================================

/**
 * 每个 allocation 使用 6 个 Int32 字段
 */
const FIELDS_PER_SLOT = 6;
const FIELD_ARENA_ID = 0;
const FIELD_OFFSET = 1;
const FIELD_SIZE = 2;
const FIELD_REF_COUNT = 3;
const FIELD_FENCE_EPOCH = 4;
const FIELD_GENERATION = 5;

// ============================================================================
// PagedAllocationStore
// ============================================================================

interface Page {
    /** 主数据数组：[arenaId, offset, size, refCount, fenceEpoch, generation] × PAGE_SIZE */
    data: Int32Array;
    /** Debug 模式下的 Token 数组 */
    debugTokens: Uint32Array | null;
}

export class PagedAllocationStore {
    private pages: Page[] = [];
    private nextId: number = 0;
    private freeHead: number = INVALID_ID;  // 空闲链表头
    private freeCount: number = 0;

    // Debug 模式统计
    private _totalAllocations: number = 0;
    private _peakAllocations: number = 0;

    constructor() {
        // 预分配第一页
        this.allocatePage();
    }

    // ========================================================================
    // Page Management
    // ========================================================================

    private allocatePage(): void {
        const page: Page = {
            data: new Int32Array(PAGE_SIZE * FIELDS_PER_SLOT),
            debugTokens: DEBUG_MODE ? new Uint32Array(PAGE_SIZE) : null,
        };

        // 初始化所有 slot 为无效状态
        for (let i = 0; i < PAGE_SIZE; i++) {
            const base = i * FIELDS_PER_SLOT;
            page.data[base + FIELD_ARENA_ID] = INVALID_ID;
            page.data[base + FIELD_GENERATION] = 0;
        }

        this.pages.push(page);
    }

    private getPage(id: number): Page {
        const pageIndex = Math.floor(id / PAGE_SIZE);
        if (pageIndex >= this.pages.length) {
            throw new Error(`Page ${pageIndex} does not exist for id ${id}`);
        }
        return this.pages[pageIndex];
    }

    private getSlotBase(id: number): number {
        return (id % PAGE_SIZE) * FIELDS_PER_SLOT;
    }

    // ========================================================================
    // ID Allocation
    // ========================================================================

    /**
     * 分配一个新的 Allocation ID
     * 
     * 时间复杂度: O(1)
     */
    allocId(): number {
        let id: number;

        if (this.freeHead !== INVALID_ID) {
            // 从空闲链表取
            id = this.freeHead;
            const page = this.getPage(id);
            const base = this.getSlotBase(id);
            // 空闲链表用 offset 字段存储 next 指针
            this.freeHead = page.data[base + FIELD_OFFSET];
            this.freeCount--;
        } else {
            // 分配新 ID
            id = this.nextId++;
            const pageIndex = Math.floor(id / PAGE_SIZE);
            if (pageIndex >= this.pages.length) {
                this.allocatePage();
            }
        }

        // 初始化 slot
        const page = this.getPage(id);
        const base = this.getSlotBase(id);
        const slotIndex = id % PAGE_SIZE;

        // 递增 generation（用于检测 use-after-free）
        const newGeneration = page.data[base + FIELD_GENERATION] + 1;

        // 注意：arenaId 初始化为 0（有效值），由调用者设置；refCount = 1 表示活跃
        page.data[base + FIELD_ARENA_ID] = 0;  // 临时值，调用者会设置
        page.data[base + FIELD_OFFSET] = 0;
        page.data[base + FIELD_SIZE] = 0;
        page.data[base + FIELD_REF_COUNT] = 1;  // 初始引用计数 = 1，表示活跃
        page.data[base + FIELD_FENCE_EPOCH] = 0;
        page.data[base + FIELD_GENERATION] = newGeneration;

        // Debug 模式：生成随机 Token
        if (DEBUG_MODE && page.debugTokens) {
            page.debugTokens[slotIndex] = (Math.random() * 0xFFFFFFFF) >>> 0;
        }

        // 统计
        this._totalAllocations++;
        this._peakAllocations = Math.max(this._peakAllocations, this._totalAllocations);

        return id;
    }

    /**
     * 分配 ID 并返回 (id, token) 对
     * 
     * Debug 模式下 token 用于校验 use-after-free
     */
    allocIdWithToken(): [number, number] {
        const id = this.allocId();
        const token = this.getDebugToken(id);
        return [id, token];
    }

    /**
     * 释放 Allocation ID
     * 
     * 时间复杂度: O(1)
     */
    freeId(id: number): void {
        if (id < 0 || id >= this.nextId) {
            throw new Error(`Invalid allocation ID: ${id}`);
        }

        const page = this.getPage(id);
        const base = this.getSlotBase(id);

        // 检查是否已释放（使用 refCount = 0 作为释放标记）
        if (page.data[base + FIELD_REF_COUNT] === 0) {
            console.warn(`Allocation ID ${id} may already be freed`);
            return;
        }

        // 标记为已释放：refCount = 0
        page.data[base + FIELD_REF_COUNT] = 0;
        page.data[base + FIELD_OFFSET] = this.freeHead;  // 复用 offset 存储链表 next

        // 加入空闲链表头
        this.freeHead = id;
        this.freeCount++;

        // 统计
        this._totalAllocations--;

        // Debug 模式：递增 generation 并更新 token（使旧 token 失效）
        if (DEBUG_MODE) {
            page.data[base + FIELD_GENERATION]++;
            // 更新 token 使旧的失效
            if (page.debugTokens) {
                page.debugTokens[id % PAGE_SIZE] = (Math.random() * 0xFFFFFFFF) >>> 0;
            }
        }
    }

    // ========================================================================
    // Field Accessors
    // ========================================================================

    getArenaId(id: number): number {
        const page = this.getPage(id);
        return page.data[this.getSlotBase(id) + FIELD_ARENA_ID];
    }

    setArenaId(id: number, arenaId: number): void {
        const page = this.getPage(id);
        page.data[this.getSlotBase(id) + FIELD_ARENA_ID] = arenaId;
    }

    getOffset(id: number): number {
        const page = this.getPage(id);
        const base = this.getSlotBase(id);
        // 检查是否已释放（refCount = 0 表示已释放）
        if (page.data[base + FIELD_REF_COUNT] === 0) {
            // console.error(`[CRITICAL] getOffset called on freed allocation ${id}!`);
            // 返回 0 而不是被污染的 free list 指针
            return 0;
        }
        return page.data[base + FIELD_OFFSET];
    }

    setOffset(id: number, offset: number): void {
        const page = this.getPage(id);
        page.data[this.getSlotBase(id) + FIELD_OFFSET] = offset;
    }

    getSize(id: number): number {
        const page = this.getPage(id);
        const base = this.getSlotBase(id);
        // 检查是否已释放
        if (page.data[base + FIELD_REF_COUNT] === 0) {
            // console.error(`[CRITICAL] getSize called on freed allocation ${id}!`);
            return 0;
        }
        return page.data[base + FIELD_SIZE];
    }

    setSize(id: number, size: number): void {
        const page = this.getPage(id);
        page.data[this.getSlotBase(id) + FIELD_SIZE] = size;
    }

    getRefCount(id: number): number {
        const page = this.getPage(id);
        return page.data[this.getSlotBase(id) + FIELD_REF_COUNT];
    }

    setRefCount(id: number, refCount: number): void {
        const page = this.getPage(id);
        page.data[this.getSlotBase(id) + FIELD_REF_COUNT] = refCount;
    }

    incrementRefCount(id: number): number {
        const page = this.getPage(id);
        const base = this.getSlotBase(id) + FIELD_REF_COUNT;
        return ++page.data[base];
    }

    decrementRefCount(id: number): number {
        const page = this.getPage(id);
        const base = this.getSlotBase(id) + FIELD_REF_COUNT;
        return --page.data[base];
    }

    getFenceEpoch(id: number): number {
        const page = this.getPage(id);
        return page.data[this.getSlotBase(id) + FIELD_FENCE_EPOCH];
    }

    setFenceEpoch(id: number, epoch: number): void {
        const page = this.getPage(id);
        page.data[this.getSlotBase(id) + FIELD_FENCE_EPOCH] = epoch;
    }

    getGeneration(id: number): number {
        const page = this.getPage(id);
        return page.data[this.getSlotBase(id) + FIELD_GENERATION];
    }

    // ========================================================================
    // Debug Token Validation
    // ========================================================================

    private getDebugToken(id: number): number {
        if (!DEBUG_MODE) return 0;
        const page = this.getPage(id);
        if (!page.debugTokens) return 0;
        return page.debugTokens[id % PAGE_SIZE];
    }

    /**
     * 校验 ID 有效性
     * 
     * 检查：
     * 1. ID 范围
     * 2. 是否已释放（arenaId === INVALID_ID）
     * 3. Debug Token 匹配
     * 
     * @returns true 如果有效，false 如果无效
     */
    validateId(id: number, token: number): boolean {
        // 范围检查
        if (id < 0 || id >= this.nextId) {
            if (DEBUG_MODE) {
                console.error(`Invalid AllocId: ${id} out of range [0, ${this.nextId})`);
            }
            return false;
        }

        const page = this.getPage(id);
        const base = this.getSlotBase(id);

        // 检查是否已释放（使用 refCount = 0 作为释放标记）
        if (page.data[base + FIELD_REF_COUNT] === 0) {
            if (DEBUG_MODE) {
                console.error(`Invalid AllocId: ${id} has been freed (use-after-free)`);
            }
            return false;
        }

        // Debug Token 校验
        if (DEBUG_MODE && page.debugTokens) {
            const expectedToken = page.debugTokens[id % PAGE_SIZE];
            if (expectedToken !== token) {
                console.error(
                    `Invalid AllocId: token mismatch for id ${id} ` +
                    `(expected ${expectedToken}, got ${token}). ` +
                    `This may indicate use-after-free or a forged ID.`
                );
                return false;
            }
        }

        return true;
    }

    /**
     * 校验并抛出异常
     */
    assertValid(id: number, token: number): void {
        if (!this.validateId(id, token)) {
            throw new InvalidAllocationError(id, 'validation failed');
        }
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    /**
     * 获取当前活跃分配数量
     */
    get activeCount(): number {
        return this._totalAllocations;
    }

    /**
     * 获取峰值分配数量
     */
    get peakCount(): number {
        return this._peakAllocations;
    }

    /**
     * 获取空闲槽位数量
     */
    get freeSlotCount(): number {
        return this.freeCount;
    }

    /**
     * 获取总页数
     */
    get pageCount(): number {
        return this.pages.length;
    }

    /**
     * 获取内存使用估算（字节）
     */
    get estimatedMemoryUsage(): number {
        const dataSize = this.pages.length * PAGE_SIZE * FIELDS_PER_SLOT * 4;  // Int32 = 4 bytes
        const tokenSize = DEBUG_MODE ? this.pages.length * PAGE_SIZE * 4 : 0;  // Uint32 = 4 bytes
        return dataSize + tokenSize;
    }

    // ========================================================================
    // Iteration (for Compaction)
    // ========================================================================

    /**
     * 迭代指定 Arena 的所有活跃 allocation
     */
    *iterateByArena(arenaId: number): Generator<{ id: number; offset: number; size: number }> {
        for (let id = 0; id < this.nextId; id++) {
            const page = this.getPage(id);
            const base = this.getSlotBase(id);

            if (page.data[base + FIELD_ARENA_ID] === arenaId &&
                page.data[base + FIELD_SIZE] > 0) {
                yield {
                    id,
                    offset: page.data[base + FIELD_OFFSET],
                    size: page.data[base + FIELD_SIZE],
                };
            }
        }
    }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let instance: PagedAllocationStore | null = null;

export function getAllocationStore(): PagedAllocationStore {
    if (!instance) {
        instance = new PagedAllocationStore();
    }
    return instance;
}

/**
 * 重置单例（仅用于测试）
 */
export function resetAllocationStore(): void {
    instance = null;
}
