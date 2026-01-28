// src/backend/webgpu/DataLayout.ts
import { DType, TensorData } from "@kandle/types";

/**
 * 计算物理存储所需的字节大小
 */
export function getPhysicalByteSize(logicalNumel: number, dtype: DType): number {
    switch (dtype) {
        case 'bool':
            // Bool 物理上是 u32，扩大 4 倍
            return logicalNumel * 4;
        case 'float32':
        case 'int32':
        case 'uint32':
            return logicalNumel * 4;
        case 'int8':
        case 'uint8':
            // Packed: 4个一组，逻辑大小 == 物理大小 (但在 Allocator 里会被 align 到 4)
            return logicalNumel * 1;
        default:
            throw new Error(`Unsupported dtype layout: ${dtype}`);
    }
}

/**
 * 将 Host 端数据编码为 Device 端兼容的物理数据
 * 例如: Uint8Array([1, 0]) -> Uint32Array([1, 0]) -> ArrayBuffer(8 bytes)
 */
export function encodeDataForStorage(data: TensorData, dtype: DType): ArrayBuffer {
    if (dtype === 'bool') {
        // === 特殊处理 Bool ===
        // Host: [1, 0] (2 bytes)
        // Device: [1, 0] (u32, 8 bytes)
        // 转换: 必须扩充
        const len = data.length;
        const expanded = new Uint32Array(len);
        for(let i=0; i<len; i++) {
            expanded[i] = data[i] ? 1 : 0;
        }
        return expanded.buffer;
    }

    // === 普通情况 / Packed 情况 ===
    // float32, int32, uint32: 直接拷贝
    // int8, uint8: 直接拷贝 (因为我们在 shader 里做了位运算解包，物理内存布局就是紧凑的字节流)

    // 确保返回的是 ArrayBuffer 且 Copy 了一份，防止外部修改影响
    return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
}

// 定义一个简单的 Type Alias 方便使用
