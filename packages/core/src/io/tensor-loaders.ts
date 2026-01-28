/**
 * Tensor I/O 加载方法
 * 
 * 提供从各种格式创建 Tensor 的工具函数
 */

import { DType, DeviceNameEnum } from '@kandle/types';
import { Tensor } from '../tensor';
import { SafetensorLayer } from './safetensor/types';
import { NpyDescriptor } from './npy/types';
import { convertBF16toF32, createTypedArrayFromBuffer } from './safetensor/dtypes';

// ============================================================================
// Types
// ============================================================================

/**
 * Tensor 加载选项
 */
export interface TensorLoadOptions {
    /** 目标设备 */
    device?: DeviceNameEnum;
    /** 目标 dtype (自动转换) */
    dtype?: DType;
}

// ============================================================================
// Tensor.fromSafetensorLayer
// ============================================================================

/**
 * 从 Safetensor Layer 创建 Tensor
 * 
 * @param layer - Layer 描述符
 * @param options - 加载选项
 * @param signal - 可选的取消信号
 * @returns 新的 Tensor（数据在目标设备上）
 * 
 * @example
 * const group = await loadSafetensor('./model.safetensors');
 * const layer = group.getLayer('model.layers.0.weight');
 * const tensor = await tensorFromSafetensorLayer(layer, { device: 'webgpu' });
 */
export async function tensorFromSafetensorLayer(
    layer: SafetensorLayer,
    options: TensorLoadOptions = {},
    signal?: AbortSignal
): Promise<Tensor> {
    const { device, dtype } = options;

    // 1. 从 ByteSource 读取原始字节
    const absoluteOffset = layer.file.dataOffset + layer.dataOffsets[0];
    const buffer = await layer.file.source.read(absoluteOffset, layer.byteSize, signal);

    // 2. 处理 BF16 → F32 转换
    let processedBuffer = buffer;
    let targetDtype = dtype ?? layer.dtype;

    if (layer.originalDtype === 'BF16') {
        processedBuffer = convertBF16toF32(buffer);
        targetDtype = dtype ?? 'float32';
    }

    // 3. 创建 TypedArray（使用 slice 确保内存对齐）
    const alignedBuffer = processedBuffer.slice(0);
    const typedArray = createTypedArrayFromBuffer(alignedBuffer, targetDtype);

    // 4. 创建 Tensor
    const tensor = new Tensor(typedArray, {
        shape: [...layer.shape],
        dtype: targetDtype,
        device,
    });

    return tensor;
}

// ============================================================================
// Tensor.fromNpy
// ============================================================================

/**
 * 从 NPY 描述符创建 Tensor
 * 
 * @param descriptor - NPY 描述符
 * @param options - 加载选项
 * @param signal - 可选的取消信号
 * @returns 新的 Tensor（数据在目标设备上）
 * 
 * @example
 * const desc = await loadNpy('./weights.npy');
 * const tensor = await tensorFromNpy(desc, { device: 'webgpu' });
 */
export async function tensorFromNpy(
    descriptor: NpyDescriptor,
    options: TensorLoadOptions = {},
    signal?: AbortSignal
): Promise<Tensor> {
    const { device, dtype } = options;

    // 1. 读取数据
    const buffer = await descriptor.source.read(descriptor.dataOffset, descriptor.byteSize, signal);

    // 2. 处理 Fortran order (需要转置)
    if (descriptor.fortranOrder) {
        throw new Error('Fortran order NPY not yet supported. Consider converting to C order in Python first.');
    }

    // 3. 处理字节序 (如果是 big endian，需要转换)
    let processedBuffer = buffer;
    if (descriptor.byteOrder === 'big') {
        processedBuffer = swapByteOrder(buffer, descriptor.dtype);
    }

    // 4. 创建 TypedArray（使用 slice 确保内存对齐）
    const alignedBuffer = processedBuffer.slice(0);
    const targetDtype = dtype ?? descriptor.dtype;
    const typedArray = createTypedArrayFromBuffer(alignedBuffer, targetDtype);

    // 5. 创建 Tensor
    const tensor = new Tensor(typedArray, {
        shape: [...descriptor.shape],
        dtype: targetDtype,
        device,
    });

    return tensor;
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * 交换字节序 (big endian → little endian)
 */
function swapByteOrder(buffer: ArrayBuffer, dtype: DType): ArrayBuffer {
    const bytesPerElement = getBytesPerElement(dtype);

    if (bytesPerElement === 1) {
        // 单字节无需交换
        return buffer;
    }

    const result = new ArrayBuffer(buffer.byteLength);
    const src = new Uint8Array(buffer);
    const dst = new Uint8Array(result);

    for (let i = 0; i < buffer.byteLength; i += bytesPerElement) {
        for (let j = 0; j < bytesPerElement; j++) {
            dst[i + j] = src[i + (bytesPerElement - 1 - j)];
        }
    }

    return result;
}

function getBytesPerElement(dtype: DType): number {
    const map: Record<DType, number> = {
        float64: 8,
        float32: 4,
        float16: 2,
        int64: 8,
        int32: 4,
        int16: 2,
        int8: 1,
        uint64: 8,
        uint32: 4,
        uint16: 2,
        uint8: 1,
        bool: 1,
        complex64: 8,
        complex128: 16,
    };
    return map[dtype] ?? 4;
}
