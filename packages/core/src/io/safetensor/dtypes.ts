/**
 * Safetensors DType 映射和转换工具
 */

import { DType, DataTypeMap } from '@kandle/types';
import { SafetensorsDType } from './types';

// ============================================================================
// DType Mapping
// ============================================================================

/**
 * Safetensors dtype → NN-Kit dtype 映射
 * 
 * 注意: BF16 映射到 float32（因为 WebGPU 不支持 BF16，需要运行时转换）
 */
export const SAFETENSORS_DTYPE_MAP: Record<SafetensorsDType, DType> = {
    F64: 'float64',
    F32: 'float32',
    F16: 'float16',
    BF16: 'float32',  // BF16 转换为 float32
    I64: 'int64',
    I32: 'int32',
    I16: 'int16',
    I8: 'int8',
    U64: 'uint64',
    U32: 'uint32',
    U16: 'uint16',
    U8: 'uint8',
    BOOL: 'bool',
};

/**
 * 各 dtype 的每元素字节数
 */
export const SAFETENSORS_DTYPE_BYTES: Record<SafetensorsDType, number> = {
    F64: 8,
    F32: 4,
    F16: 2,
    BF16: 2,
    I64: 8,
    I32: 4,
    I16: 2,
    I8: 1,
    U64: 8,
    U32: 4,
    U16: 2,
    U8: 1,
    BOOL: 1,
};

/**
 * 将 Safetensors dtype 映射到 NN-Kit dtype
 * 
 * @param stDtype - Safetensors dtype 字符串
 * @returns NN-Kit dtype
 * @throws 如果 dtype 不支持
 */
export function mapSafetensorsDType(stDtype: SafetensorsDType): DType {
    const mapped = SAFETENSORS_DTYPE_MAP[stDtype];
    if (!mapped) {
        throw new Error(`Unsupported safetensors dtype: ${stDtype}`);
    }
    return mapped;
}

// ============================================================================
// BF16 Conversion
// ============================================================================

/**
 * 将 BF16 数据转换为 F32
 * 
 * BF16 格式: [sign(1) | exponent(8) | mantissa(7)]
 * F32 格式:  [sign(1) | exponent(8) | mantissa(23)]
 * 
 * 转换方式: BF16 值左移 16 位即为 F32
 * 
 * @param buffer - BF16 数据的 ArrayBuffer
 * @returns F32 数据的 ArrayBuffer
 */
export function convertBF16toF32(buffer: ArrayBuffer): ArrayBuffer {
    const bf16 = new Uint16Array(buffer);
    const f32Buffer = new ArrayBuffer(bf16.length * 4);
    const f32 = new Float32Array(f32Buffer);
    const view = new DataView(f32Buffer);

    for (let i = 0; i < bf16.length; i++) {
        // BF16 值作为 F32 的高 16 位，低 16 位补 0
        view.setUint32(i * 4, bf16[i] << 16, true);
    }

    return f32Buffer;
}

// ============================================================================
// TypedArray Creation
// ============================================================================

/**
 * NN-Kit dtype → TypedArray 构造函数
 */
const DTYPE_TO_TYPED_ARRAY: Record<DType, new (buffer: ArrayBuffer) => DataTypeMap[DType]> = {
    float64: Float64Array,
    float32: Float32Array,
    float16: Uint16Array as any,  // float16 存储为 uint16
    int64: BigInt64Array as any,
    int32: Int32Array,
    int16: Int16Array,
    int8: Int8Array,
    uint64: BigUint64Array as any,
    uint32: Uint32Array,
    uint16: Uint16Array,
    uint8: Uint8Array,
    bool: Uint8Array,
    complex64: Float32Array,
    complex128: Float64Array,
};

/**
 * 从 ArrayBuffer 创建对应类型的 TypedArray
 * 
 * @param buffer - 原始字节数据
 * @param dtype - 目标 dtype
 * @returns 对应类型的 TypedArray
 */
export function createTypedArrayFromBuffer(buffer: ArrayBuffer, dtype: DType): DataTypeMap[DType] {
    const TypedArrayCtor = DTYPE_TO_TYPED_ARRAY[dtype];
    if (!TypedArrayCtor) {
        throw new Error(`Unsupported dtype for TypedArray creation: ${dtype}`);
    }
    return new TypedArrayCtor(buffer);
}

/**
 * 获取 dtype 的每元素字节数
 */
export function getDtypeBytes(dtype: DType): number {
    const bytesMap: Record<DType, number> = {
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
        complex64: 8,   // 2 * float32
        complex128: 16, // 2 * float64
    };
    return bytesMap[dtype] ?? 4;
}
