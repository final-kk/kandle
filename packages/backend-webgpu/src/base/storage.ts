import { DeviceNameEnum, IStorage } from "@kandle/types";
import { WebGPUAllocator } from "./allocator";
import { WebGPUDeviceManager } from "./device";
import { GlobalIdManager } from "@kandle/utils";

const storageRegistry = new FinalizationRegistry((buffer: GPUBuffer) => {
    // 仅在 buffer 未被手动释放时销毁
    // 注意：GPUBuffer 被销毁后访问 mappedState 会抛出异常
    try {
        buffer.destroy();
    } catch {
        // Buffer 已被销毁，忽略
    }
})

// 全局统计：用于诊断显存泄漏
let _totalStorageCreated = 0;
let _totalStorageDestroyed = 0;
let _activeStorageCount = 0;
let _totalBytesAllocated = 0;
let _totalBytesFreed = 0;

// 每 100 次操作打印一次统计
let _operationCount = 0;
const LOG_INTERVAL = 100;

function maybeLogStats(operation: string, storageId: number, byteLength: number, refCount: number) {
    _operationCount++;
    if (_operationCount % LOG_INTERVAL === 0) {
        console.log(`[Storage Stats] active=${_activeStorageCount}, created=${_totalStorageCreated}, destroyed=${_totalStorageDestroyed}, bytesLive=${(_totalBytesAllocated - _totalBytesFreed) / 1024 / 1024}MB`);
    }
}

/**
 * 获取 Storage 统计信息 - 用于诊断显存泄漏
 * 
 * 调用方式: import { getStorageStats } from '@kandle/backend-webgpu'
 */
export function getStorageStats() {
    return {
        activeCount: _activeStorageCount,
        totalCreated: _totalStorageCreated,
        totalDestroyed: _totalStorageDestroyed,
        leakedCount: _totalStorageCreated - _totalStorageDestroyed,
        bytesAllocated: _totalBytesAllocated,
        bytesFreed: _totalBytesFreed,
        bytesLive: _totalBytesAllocated - _totalBytesFreed,
        bytesLiveMB: (_totalBytesAllocated - _totalBytesFreed) / 1024 / 1024,
    };
}

/**
 * 立即打印详细的 Storage 统计
 */
export function logStorageStats(label = 'Storage Stats') {
    const stats = getStorageStats();
    console.log(`[${label}] active=${stats.activeCount}, leaked=${stats.leakedCount}, bytesLive=${stats.bytesLiveMB.toFixed(2)}MB`);
}

export class WebGPUStorage implements IStorage {

    readonly storageId: number;
    readonly _byteLength: number;
    readonly _buffer: GPUBuffer;
    private _disposed: boolean = false;
    private _refCount: number = 1;

    constructor(arg: number | ArrayBuffer | GPUBuffer) {

        this.storageId = GlobalIdManager.getNextStorageId();

        if (typeof arg === "number") {

            this._byteLength = arg;

            this._buffer = WebGPUAllocator.alloc(arg);

        } else {

            if (arg instanceof ArrayBuffer) {

                this._byteLength = arg.byteLength;

                this._buffer = WebGPUAllocator.alloc(this._byteLength);

                this.upload(arg);

            } else {

                this._byteLength = (arg as GPUBuffer).size;

                this._buffer = arg as GPUBuffer;

            }

        }

        // 统计
        _totalStorageCreated++;
        _activeStorageCount++;
        _totalBytesAllocated += this._byteLength;
        // maybeLogStats('CREATE', this.storageId, this._byteLength, this._refCount);

        storageRegistry.register(this, this._buffer, this);

    }

    /**
     * 增加引用计数
     * 
     * 当创建共享此 storage 的 tensor view 时调用。
     * 例如：transpose、slice、reshape 等 view 操作。
     */
    incRef(): void {
        if (this._disposed) {
            throw new Error(`Cannot incRef a disposed storage (id=${this.storageId})`);
        }
        this._refCount++;
        // maybeLogStats('INCREF', this.storageId, this._byteLength, this._refCount);
    }

    /**
     * 减少引用计数
     * 
     * 当 tensor/view dispose 时调用。
     * 引用计数归零时销毁 GPUBuffer。
     */
    decRef(): void {
        if (this._disposed) {
            return;
        }
        this._refCount--;
        // maybeLogStats('DECREF', this.storageId, this._byteLength, this._refCount);
        if (this._refCount <= 0) {
            this._disposed = true;
            storageRegistry.unregister(this);
            this._buffer.destroy();
            // 统计
            _totalStorageDestroyed++;
            _activeStorageCount--;
            _totalBytesFreed += this._byteLength;
        }
    }

    /**
     * 释放此 tensor 对 storage 的引用
     * 
     * 对于拥有独立 storage 的 tensor，这将导致 buffer 立即销毁。
     * 对于共享 storage 的 view，这只是减少引用计数。
     */
    dispose(): void {
        this.decRef();
    }

    /**
     * 检查此 Storage 是否已被释放
     */
    get isDisposed(): boolean {
        return this._disposed;
    }

    /**
     * 当前引用计数（用于调试）
     */
    get refCount(): number {
        return this._refCount;
    }

    get device(): DeviceNameEnum {
        return DeviceNameEnum.WebGPU;
    }

    get buffer() {
        return this._buffer
    }

    get byteLength(): number {
        return this._byteLength;
    }

    /**
     * 数据在 buffer 中的字节偏移量
     * 
     * 由于不再使用 Arena-based Memory Pool，每个 Storage 独占一个 buffer，
     * 因此 bufferOffset 始终为 0。
     */
    get bufferOffset(): number {
        return 0;
    }

    async toRawDataAsync(): Promise<ArrayBuffer> {

        const device = WebGPUDeviceManager.device;

        const size = this._byteLength;

        // Align size to 4 bytes as required by WebGPU spec
        // If data size is not aligned (e.g. 6 bytes), we must round up.
        // We assume the underlying buffer is large enough or padded.
        const alignedSize = (size + 3) & ~3;

        const stagingBuffer = device.createBuffer({
            size: alignedSize,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        const commandEncoder = device.createCommandEncoder();

        commandEncoder.copyBufferToBuffer(
            this._buffer, 0,
            stagingBuffer, 0,
            alignedSize // We assumed source buffer is aligned/padded in Allocator.
        );

        device.queue.submit([commandEncoder.finish()]);

        await stagingBuffer.mapAsync(GPUMapMode.READ, 0, alignedSize);

        const srcArrayBuffer = stagingBuffer.getMappedRange(0, alignedSize);

        const resultArrayBuffer = srcArrayBuffer.slice(0, size);

        stagingBuffer.unmap();

        stagingBuffer.destroy();

        return resultArrayBuffer;

    }

    upload(data: ArrayBuffer): void {

        const device = WebGPUDeviceManager.device;

        device.queue.writeBuffer(
            this._buffer,
            0,
            data,
            0,
            data.byteLength
        );

    }

}
