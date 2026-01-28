/**
 * PooledWebGPUStorage
 *
 * 使用 GPU Memory Pool 的 WebGPU Storage 实现。
 *
 * 设计要点：
 * 1. 持有 allocId 而非直接持有 GPUBuffer
 * 2. 通过 Memory Pool 子模块动态获取 buffer 和 offset
 * 3. Debug 模式下进行 Token 校验
 * 4. 使用 FinalizationRegistry 兜底防泄漏
 * 5. 引用计数支持多 Tensor 共享 Storage
 */

/// <reference types="@webgpu/types" />

import { DeviceNameEnum, IStorage } from "@kandle/types";
import { GlobalIdManager } from "@kandle/utils";
import { BufferUsagePreset, DEBUG_MODE, InvalidAllocationError } from "./types";
import { getArenaManager } from "./arena-manager";
import { getStagingPool } from "./staging-pool";
import { getAllocationStore } from "./allocation-store";
import { getFenceTracker, getDeferredReleaseQueue } from "./fence-tracker";

// ============================================================================
// FinalizationRegistry for leak prevention
// ============================================================================

const storageRegistry = new FinalizationRegistry(
    (heldValue: { allocId: number; debugToken: number }) => {
        // 兜底：如果 Storage 被 GC 但没有调用 dispose，尝试释放
        try {
            const store = getAllocationStore();
            const newRefCount = store.decrementRefCount(heldValue.allocId);
            if (newRefCount === 0) {
                // 直接释放到 Free List
                getArenaManager().free(heldValue.allocId);
            }
        } catch (e) {
            // 内存池可能已经销毁
        }
    }
);

// ============================================================================
// PooledWebGPUStorage
// ============================================================================

export class PooledWebGPUStorage implements IStorage {
    readonly storageId: number;
    readonly _byteLength: number;

    /** Allocation ID */
    private readonly _allocId: number;

    /** Debug Token (用于校验 use-after-free) */
    private readonly _debugToken: number;

    /** 是否已释放 */
    private _disposed = false;

    constructor(arg: number | ArrayBuffer, usage?: GPUBufferUsageFlags) {
        this.storageId = GlobalIdManager.getNextStorageId();

        const arenaManager = getArenaManager();
        const effectiveUsage = usage ?? BufferUsagePreset.STORAGE;

        if (typeof arg === "number") {
            // 纯分配
            this._byteLength = arg;
            const result = arenaManager.allocate(arg, effectiveUsage);
            this._allocId = result.allocId;
            this._debugToken = result.debugToken;
        } else {
            // 分配 + 上传
            this._byteLength = arg.byteLength;
            const result = arenaManager.allocate(arg.byteLength, effectiveUsage);
            this._allocId = result.allocId;
            this._debugToken = result.debugToken;

            // 上传数据
            const buffer = arenaManager.getBuffer(this._allocId);
            const offset = arenaManager.getOffset(this._allocId);
            arenaManager.device.queue.writeBuffer(buffer, offset, arg);
        }

        // 注册 GC 兜底
        storageRegistry.register(
            this,
            { allocId: this._allocId, debugToken: this._debugToken },
            this // unregister token
        );
    }

    // ========================================================================
    // IStorage Implementation
    // ========================================================================

    get device(): DeviceNameEnum {
        return DeviceNameEnum.WebGPU;
    }

    get byteLength(): number {
        return this._byteLength;
    }

    /**
     * 获取 GPUBuffer
     *
     * 注意：返回的是 Arena 的整个 buffer，需要配合 bufferOffset 使用
     */
    get buffer(): GPUBuffer {
        this.checkValid();
        return getArenaManager().getBuffer(this._allocId);
    }

    /**
     * 获取在 buffer 中的偏移量
     */
    get bufferOffset(): number {
        this.checkValid();
        return getArenaManager().getOffset(this._allocId);
    }

    /**
     * 获取 Allocation ID
     */
    get allocId(): number {
        return this._allocId;
    }

    /**
     * 异步读取数据
     */
    async toRawDataAsync(): Promise<ArrayBuffer> {
        this.checkValid();
        const arenaManager = getArenaManager();
        const buffer = arenaManager.getBuffer(this._allocId);
        const offset = arenaManager.getOffset(this._allocId);
        return getStagingPool().readFromBuffer(buffer, offset, this._byteLength);
    }

    /**
     * 上传数据
     */
    upload(data: ArrayBuffer): void {
        this.checkValid();
        const arenaManager = getArenaManager();
        const buffer = arenaManager.getBuffer(this._allocId);
        const offset = arenaManager.getOffset(this._allocId);
        arenaManager.device.queue.writeBuffer(buffer, offset, data);
    }

    // ========================================================================
    // Reference Counting
    // ========================================================================

    /**
     * 增加引用计数
     * 
     * 用于 createView 时共享 Storage
     * 
     * 注意：不检查 _disposed，因为多个 Tensor 可能共享同一个 Storage 实例。
     * 即使某个 Tensor 已调用 dispose()，其他 Tensor 仍可能持有有效引用。
     */
    addRef(): void {
        const store = getAllocationStore();
        const refCount = store.getRefCount(this._allocId);

        if (refCount === 0) {
            throw new Error(`Cannot addRef on freed allocation ${this._allocId} (refCount=0)`);
        }

        store.incrementRefCount(this._allocId);
    }

    // ========================================================================
    // Lifecycle
    // ========================================================================

    /**
     * 释放引用 (refCount--)
     * 
     * 如果 refCount 归零，内存将被延迟释放
     */
    dispose(): void {
        if (this._disposed) {
            return;
        }
        this._disposed = true;

        // ⚠️ 关键：取消 GC 注册，避免 Double Decrement
        storageRegistry.unregister(this);

        // refCount--
        const store = getAllocationStore();
        const newRefCount = store.decrementRefCount(this._allocId);

        if (newRefCount === 0) {
            // refCount 归零，直接释放到 Free List（立即可复用）
            // 对于顺序推理场景，不需要等待 GPU fence
            getArenaManager().free(this._allocId);
        } else if (newRefCount < 0) {
            // 恢复到 0，避免进一步腐败
            store.setRefCount(this._allocId, 0);
        }
    }

    /**
     * 检查底层 allocation 是否已被释放（refCount = 0）
     * 
     * 注意：这检查的是底层 allocation 状态，而非当前 dispose 调用状态
     */
    get isDisposed(): boolean {
        const store = getAllocationStore();
        return store.getRefCount(this._allocId) === 0;
    }

    // ========================================================================
    // Validation
    // ========================================================================

    private checkValid(): void {
        // 检查底层 allocation 是否仍然有效（refCount > 0）
        // 注意：不检查 _disposed，因为多个 Tensor 可能共享同一个 Storage 实例
        // 即使某个 Tensor 已调用 dispose()，只要 refCount > 0，allocation 仍然有效
        const store = getAllocationStore();
        const refCount = store.getRefCount(this._allocId);

        if (refCount === 0) {
            throw new InvalidAllocationError(
                this._allocId,
                'allocation has been freed (refCount=0)'
            );
        }

        if (DEBUG_MODE) {
            if (!store.validateId(this._allocId, this._debugToken)) {
                throw new InvalidAllocationError(
                    this._allocId,
                    'validation failed (possible use-after-free)'
                );
            }
        }
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * 创建 PooledWebGPUStorage
 */
export function createPooledStorage(
    sizeOrData: number | ArrayBuffer,
    usage?: GPUBufferUsageFlags
): PooledWebGPUStorage {
    return new PooledWebGPUStorage(sizeOrData, usage);
}
