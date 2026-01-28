/**
 * FenceTracker
 * 
 * 确定性 GPU Fence 追踪器。
 * 
 * 设计要点：
 * 1. 每次 submit 返回一个 epoch
 * 2. 使用 onSubmittedWorkDone 确认完成
 * 3. 不猜测，只使用确认完成的信号
 */

// ============================================================================
// FenceTracker
// ============================================================================

export class FenceTracker {
    private device: GPUDevice;

    /** 当前提交的 epoch */
    private currentEpoch: number = 0;

    /** 已确认完成的最大 epoch */
    private confirmedEpoch: number = 0;

    /** 正在等待确认的 epoch 列表 */
    private pendingConfirmations: Set<number> = new Set();

    constructor(device: GPUDevice) {
        this.device = device;
    }

    // ========================================================================
    // Epoch Management
    // ========================================================================

    /**
     * 提交命令并返回 epoch
     * 
     * 调用者负责实际的 queue.submit()，此方法只追踪 epoch
     */
    submit(): number {
        this.currentEpoch++;
        const epoch = this.currentEpoch;

        // 异步等待确认
        this.pendingConfirmations.add(epoch);
        this.device.queue.onSubmittedWorkDone().then(() => {
            this.pendingConfirmations.delete(epoch);
            this.confirmedEpoch = Math.max(this.confirmedEpoch, epoch);
        });

        return epoch;
    }

    /**
     * 包装 submit：实际提交命令并追踪
     */
    submitCommands(commandBuffers: GPUCommandBuffer[]): number {
        this.device.queue.submit(commandBuffers);
        return this.submit();
    }

    /**
     * 检查 epoch 是否已确认完成
     * 
     * 这是确定性的：只有当我们收到 onSubmittedWorkDone 回调后才返回 true
     */
    isConfirmedComplete(epoch: number): boolean {
        return epoch <= this.confirmedEpoch;
    }

    /**
     * 强制等待指定 epoch 完成
     */
    async waitForEpoch(epoch: number): Promise<void> {
        if (this.isConfirmedComplete(epoch)) {
            return;
        }

        // 等待 GPU 完成
        await this.device.queue.onSubmittedWorkDone();

        // 更新 confirmedEpoch（所有之前的都完成了）
        this.confirmedEpoch = Math.max(this.confirmedEpoch, this.currentEpoch);
        this.pendingConfirmations.clear();
    }

    /**
     * 强制同步：等待所有提交的命令完成
     */
    async forceSync(): Promise<void> {
        await this.waitForEpoch(this.currentEpoch);
    }

    // ========================================================================
    // Getters
    // ========================================================================

    /**
     * 获取当前 epoch
     */
    get epoch(): number {
        return this.currentEpoch;
    }

    /**
     * 获取已确认的 epoch
     */
    get confirmed(): number {
        return this.confirmedEpoch;
    }

    /**
     * 获取待确认的 epoch 数量
     */
    get pendingCount(): number {
        return this.pendingConfirmations.size;
    }
}

// ============================================================================
// DeferredReleaseQueue
// ============================================================================

interface PendingRelease {
    /** Allocation ID */
    allocId: number;
    /** 最后使用的 epoch */
    fenceEpoch: number;
}

/**
 * 延迟释放队列
 * 
 * 设计要点：
 * 1. dispose 时入队而非立即释放
 * 2. 定期检查哪些已确认完成可以释放
 * 3. OOM 时可强制清空
 */
export class DeferredReleaseQueue {
    private fenceTracker: FenceTracker;
    private releaseCallback: (allocId: number) => void;

    /** 待释放队列 */
    private pending: PendingRelease[] = [];

    constructor(
        fenceTracker: FenceTracker,
        releaseCallback: (allocId: number) => void
    ) {
        this.fenceTracker = fenceTracker;
        this.releaseCallback = releaseCallback;
    }

    // ========================================================================
    // Core Operations
    // ========================================================================

    /**
     * 入队：标记 allocation 待释放
     */
    enqueue(allocId: number, fenceEpoch?: number): void {
        const epoch = fenceEpoch ?? this.fenceTracker.epoch;
        this.pending.push({ allocId, fenceEpoch: epoch });
    }

    /**
     * Tick：处理已确认完成的释放请求
     * 
     * 建议每帧调用一次
     */
    tick(): void {
        if (this.pending.length === 0) return;

        const stillPending: PendingRelease[] = [];

        for (const item of this.pending) {
            if (this.fenceTracker.isConfirmedComplete(item.fenceEpoch)) {
                // 可以安全释放
                this.releaseCallback(item.allocId);
            } else {
                stillPending.push(item);
            }
        }

        this.pending = stillPending;
    }

    /**
     * 强制清空：等待 GPU 完成后释放所有
     * 
     * 用于 OOM 恢复
     */
    async flushAll(): Promise<void> {
        if (this.pending.length === 0) return;

        // 等待所有 GPU 命令完成
        await this.fenceTracker.forceSync();

        // 释放所有
        for (const item of this.pending) {
            this.releaseCallback(item.allocId);
        }

        this.pending = [];
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    /**
     * 获取待释放队列长度
     */
    get length(): number {
        return this.pending.length;
    }

    /**
     * 检查队列是否为空
     */
    get isEmpty(): boolean {
        return this.pending.length === 0;
    }
}

// ============================================================================
// Singleton Instances
// ============================================================================

let fenceTrackerInstance: FenceTracker | null = null;
let deferredQueueInstance: DeferredReleaseQueue | null = null;

export function getFenceTracker(): FenceTracker {
    if (!fenceTrackerInstance) {
        throw new Error('FenceTracker not initialized. Call initializeFenceTracker(device) first.');
    }
    return fenceTrackerInstance;
}

export function initializeFenceTracker(device: GPUDevice): FenceTracker {
    if (fenceTrackerInstance) {
        console.warn('FenceTracker already initialized');
        return fenceTrackerInstance;
    }
    fenceTrackerInstance = new FenceTracker(device);
    return fenceTrackerInstance;
}

export function getDeferredReleaseQueue(): DeferredReleaseQueue {
    if (!deferredQueueInstance) {
        throw new Error('DeferredReleaseQueue not initialized.');
    }
    return deferredQueueInstance;
}

export function initializeDeferredReleaseQueue(
    fenceTracker: FenceTracker,
    releaseCallback: (allocId: number) => void
): DeferredReleaseQueue {
    if (deferredQueueInstance) {
        console.warn('DeferredReleaseQueue already initialized');
        return deferredQueueInstance;
    }
    deferredQueueInstance = new DeferredReleaseQueue(fenceTracker, releaseCallback);
    return deferredQueueInstance;
}

/**
 * 重置所有单例（仅用于测试）
 */
export function resetFenceTracker(): void {
    fenceTrackerInstance = null;
    deferredQueueInstance = null;
}
