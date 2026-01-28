/**
 * GPU Buffer 追踪器
 * 
 * 用于调试：追踪所有 createBuffer 调用并提供统计和清理功能
 */

let originalCreateBuffer: GPUDevice['createBuffer'] | null = null;
let trackedBuffers: GPUBuffer[] = [];
let totalCreated = 0;

/**
 * 开始追踪所有 buffer 创建
 */
export function startBufferTracking(device: GPUDevice): void {
    if (originalCreateBuffer) {
        console.warn('Buffer tracking already started');
        return;
    }

    originalCreateBuffer = device.createBuffer.bind(device);
    trackedBuffers = [];
    totalCreated = 0;

    device.createBuffer = function (descriptor: GPUBufferDescriptor): GPUBuffer {
        const buffer = originalCreateBuffer!(descriptor);
        trackedBuffers.push(buffer);
        totalCreated++;
        return buffer;
    };

    console.log('[BufferTracker] Started tracking');
}

/**
 * 停止追踪
 */
export function stopBufferTracking(device: GPUDevice): void {
    if (originalCreateBuffer) {
        device.createBuffer = originalCreateBuffer;
        originalCreateBuffer = null;
    }
    console.log('[BufferTracker] Stopped tracking');
}

/**
 * 获取统计信息
 */
export function getBufferStats(): { total: number; alive: number; leaked: number } {
    const alive = trackedBuffers.length;
    return {
        total: totalCreated,
        alive,
        leaked: alive,  // All unmanaged buffers are considered leaks
    };
}

/**
 * 销毁所有追踪的 buffer
 */
export function destroyAllTrackedBuffers(): number {
    let destroyed = 0;
    for (const buffer of trackedBuffers) {
        try {
            buffer.destroy();
            destroyed++;
        } catch {
            // Already destroyed
        }
    }
    trackedBuffers = [];
    console.log(`[BufferTracker] Destroyed ${destroyed} buffers`);
    return destroyed;
}

/**
 * 仅销毁小尺寸 buffer (likely uniform buffers)
 */
export function destroySmallBuffers(maxSize: number = 16384): number {
    let destroyed = 0;
    const remaining: GPUBuffer[] = [];

    for (const buffer of trackedBuffers) {
        try {
            if (buffer.size <= maxSize) {
                buffer.destroy();
                destroyed++;
            } else {
                remaining.push(buffer);
            }
        } catch {
            // Already destroyed
        }
    }

    trackedBuffers = remaining;
    console.log(`[BufferTracker] Destroyed ${destroyed} small buffers (size <= ${maxSize})`);
    return destroyed;
}
