/**
 * LeakDetector
 * 
 * Debug 模式下的内存泄漏检测器。
 * 
 * 功能：
 * 1. 追踪所有活跃分配的创建时间和调用栈
 * 2. 检测长时间未释放的"潜在泄漏"
 * 3. 提供 snapshot 和 reportLeaks API
 */

import { DEBUG_MODE } from './types';

// ============================================================================
// Types
// ============================================================================

export interface AllocationDebugInfo {
    /** Allocation ID */
    id: number;
    /** 分配的字节大小 */
    size: number;
    /** 创建时的 performance.now() */
    createdAt: number;
    /** 创建时的帧计数 */
    createdFrame: number;
    /** 创建时的调用栈 */
    stackTrace: string;
}

// ============================================================================
// LeakDetector
// ============================================================================

class LeakDetectorImpl {
    private allocations: Map<number, AllocationDebugInfo> = new Map();
    private frameCount: number = 0;

    /**
     * 每帧开始时调用，推进帧计数器
     */
    tick(): void {
        this.frameCount++;
    }

    /**
     * 追踪新分配
     */
    track(id: number, size: number): void {
        if (!DEBUG_MODE) return;

        this.allocations.set(id, {
            id,
            size,
            createdAt: performance.now(),
            createdFrame: this.frameCount,
            stackTrace: new Error().stack ?? '',
        });
    }

    /**
     * 取消追踪（释放时调用）
     */
    untrack(id: number): void {
        this.allocations.delete(id);
    }

    /**
     * 获取当前所有存活分配的快照
     */
    snapshot(): AllocationDebugInfo[] {
        return Array.from(this.allocations.values());
    }

    /**
     * 获取当前帧计数
     */
    get currentFrame(): number {
        return this.frameCount;
    }

    /**
     * 检测并报告潜在泄漏
     * 
     * @param thresholdFrames 超过此帧数的分配被视为潜在泄漏
     */
    reportLeaks(thresholdFrames: number = 1000): AllocationDebugInfo[] {
        const currentFrame = this.frameCount;
        const leaks = this.snapshot().filter(
            a => currentFrame - a.createdFrame > thresholdFrames
        );

        if (leaks.length > 0) {
            console.warn(`[LeakDetector] ${leaks.length} potential leaks detected:`);
            console.table(leaks.map(l => ({
                id: l.id,
                size: `${(l.size / 1024).toFixed(1)} KB`,
                age: `${currentFrame - l.createdFrame} frames`,
                createdAt: `${((performance.now() - l.createdAt) / 1000).toFixed(1)}s ago`,
                stackTrace: l.stackTrace.split('\n').slice(2, 5).join('\n'),
            })));
        }

        return leaks;
    }

    /**
     * 获取统计信息
     */
    getStats(): {
        activeCount: number;
        totalBytes: number;
        oldestFrame: number;
        newestFrame: number;
    } {
        const allocations = this.snapshot();

        if (allocations.length === 0) {
            return {
                activeCount: 0,
                totalBytes: 0,
                oldestFrame: 0,
                newestFrame: 0,
            };
        }

        const totalBytes = allocations.reduce((sum, a) => sum + a.size, 0);
        const frames = allocations.map(a => a.createdFrame);

        return {
            activeCount: allocations.length,
            totalBytes,
            oldestFrame: Math.min(...frames),
            newestFrame: Math.max(...frames),
        };
    }

    /**
     * 清空所有追踪数据
     */
    clear(): void {
        this.allocations.clear();
        this.frameCount = 0;
    }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let instance: LeakDetectorImpl | null = null;

export function getLeakDetector(): LeakDetectorImpl {
    if (!instance) {
        instance = new LeakDetectorImpl();
    }
    return instance;
}

/**
 * 重置单例（仅用于测试）
 */
export function resetLeakDetector(): void {
    instance = null;
}

// ============================================================================
// Public Debug API
// ============================================================================

/**
 * Debug 命名空间，提供内存调试功能
 */
export namespace debug {
    /**
     * 获取当前所有存活分配的快照
     */
    export function snapshot(): AllocationDebugInfo[] {
        return getLeakDetector().snapshot();
    }

    /**
     * 检测并报告潜在泄漏
     */
    export function reportLeaks(thresholdFrames?: number): AllocationDebugInfo[] {
        return getLeakDetector().reportLeaks(thresholdFrames);
    }

    /**
     * 获取内存统计信息
     */
    export function stats(): ReturnType<LeakDetectorImpl['getStats']> {
        return getLeakDetector().getStats();
    }

    /**
     * 推进帧计数器（每帧调用）
     */
    export function tick(): void {
        getLeakDetector().tick();
    }
}
