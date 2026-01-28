/**
 * DType Resolver - 物理类型与逻辑类型分离架构
 * 
 * ## 设计目标
 * 1. 单一职责: 每个函数只做一件事
 * 2. 消除运行时分支: 在初始化时确定策略，运行时直接执行
 * 3. 透明性: 用户能明确知道 float16 的实际存储/计算精度
 * 4. 可测试性: 支持模拟不同设备能力进行测试
 * 
 * ## 核心思想
 * 将逻辑类型 (DType) 与物理存储类型 (WGSL Type) 解耦，
 * 在 Backend 初始化时根据设备能力一次性确定所有类型的处理策略。
 */

import { DType } from "@kandle/types";
import { WgslDType } from "../types";

// ============================================================
// 类型定义
// ============================================================

/**
 * TypedArray 构造函数类型
 */
export type TypedArrayConstructor =
    | Uint8ArrayConstructor
    | Int8ArrayConstructor
    | Uint16ArrayConstructor
    | Int16ArrayConstructor
    | Uint32ArrayConstructor
    | Int32ArrayConstructor
    | Float32ArrayConstructor
    | Float64ArrayConstructor
    | BigInt64ArrayConstructor
    | BigUint64ArrayConstructor;

/**
 * 类型转换函数签名
 * 
 * @param src 源数据 (ArrayBuffer)
 * @param numel 逻辑元素数量
 * @returns 转换后的 ArrayBuffer
 */
export type DataConverter = (src: ArrayBuffer, numel: number) => ArrayBuffer;

/**
 * 物理存储描述符
 * 
 * 在 Backend 初始化时确定，运行时不变。
 * 描述了一种逻辑 dtype 如何在 GPU 上存储和计算。
 */
export interface PhysicalStorageDescriptor {
    /** 逻辑类型 */
    readonly logicalDType: DType;

    /** GPU 端 WGSL 存储类型 (e.g. 'f16', 'f32', 'i32') */
    readonly wgslStorageType: WgslDType;

    /** GPU 端 WGSL 计算类型 (可能与存储类型不同，如 f16 存储但 f32 计算) */
    readonly wgslComputeType: WgslDType;

    /** 每个逻辑元素在 GPU 中占用的字节数 */
    readonly gpuBytesPerElement: number;

    /** JS 端返回给用户的 TypedArray 构造函数 */
    readonly jsTypedArrayCtor: TypedArrayConstructor;

    /** 上传时是否需要转换 */
    readonly needsUploadConversion: boolean;

    /** 下载时是否需要转换 */
    readonly needsDownloadConversion: boolean;

    /** 上传时的转换函数 (JS → GPU) */
    readonly uploadConverter: DataConverter;

    /** 下载时的转换函数 (GPU → JS) */
    readonly downloadConverter: DataConverter;
}

/**
 * DType 解析器
 * 
 * 在 Backend 初始化时构建，包含所有类型的物理存储描述。
 * 运行时通过查表 O(1) 获取描述符，无需条件判断。
 */
export interface IDTypeResolver {
    /** 获取指定 dtype 的物理存储描述符 */
    getDescriptor(dtype: DType): PhysicalStorageDescriptor;

    /** 当前设备是否原生支持 float16 */
    readonly supportsNativeF16: boolean;

    /** float16 的实际存储精度 (用于用户查询/警告) */
    readonly float16StoragePrecision: 'f16' | 'f32';

    /** 计算指定 dtype 和元素数量所需的 GPU 存储字节数 */
    calculateStorageBytes(dtype: DType, numel: number): number;
}

// ============================================================
// Float16 转换函数
// ============================================================

/**
 * 将 IEEE 754 半精度浮点数 (16-bit) 转换为单精度浮点数 (32-bit)
 */
export function float16ToFloat32(h: number): number {
    const sign = (h >>> 15) & 0x1;
    const exponent = (h >>> 10) & 0x1F;
    const mantissa = h & 0x3FF;

    let f32: number;

    if (exponent === 0) {
        if (mantissa === 0) {
            f32 = sign === 0 ? 0 : -0;
        } else {
            // 非规格化数 (subnormal)
            let e = -14;
            let m = mantissa;
            while ((m & 0x400) === 0) {
                m <<= 1;
                e--;
            }
            m &= 0x3FF;
            const f32Exp = e + 127;
            const f32Mantissa = m << 13;
            const bits = (sign << 31) | (f32Exp << 23) | f32Mantissa;
            const buffer = new ArrayBuffer(4);
            new Uint32Array(buffer)[0] = bits;
            f32 = new Float32Array(buffer)[0];
        }
    } else if (exponent === 0x1F) {
        // Infinity 或 NaN
        if (mantissa === 0) {
            f32 = sign === 0 ? Infinity : -Infinity;
        } else {
            f32 = NaN;
        }
    } else {
        // 规格化数
        const f32Exp = exponent - 15 + 127;
        const f32Mantissa = mantissa << 13;
        const bits = (sign << 31) | (f32Exp << 23) | f32Mantissa;
        const buffer = new ArrayBuffer(4);
        new Uint32Array(buffer)[0] = bits;
        f32 = new Float32Array(buffer)[0];
    }

    return f32;
}

/**
 * 将单精度浮点数 (32-bit) 转换为 IEEE 754 半精度浮点数 (16-bit)
 */
export function float32ToFloat16(f: number): number {
    const buffer = new ArrayBuffer(4);
    new Float32Array(buffer)[0] = f;
    const bits = new Uint32Array(buffer)[0];

    const sign = (bits >>> 31) & 0x1;
    const exponent = (bits >>> 23) & 0xFF;
    const mantissa = bits & 0x7FFFFF;

    let h: number;

    if (exponent === 0) {
        h = sign << 15;
    } else if (exponent === 0xFF) {
        if (mantissa === 0) {
            h = (sign << 15) | 0x7C00;
        } else {
            h = (sign << 15) | 0x7C00 | (mantissa >>> 13);
        }
    } else {
        let newExp = exponent - 127 + 15;
        if (newExp >= 0x1F) {
            h = (sign << 15) | 0x7C00;
        } else if (newExp <= 0) {
            if (newExp < -10) {
                h = sign << 15;
            } else {
                const m = (mantissa | 0x800000) >>> (1 - newExp);
                h = (sign << 15) | (m >>> 13);
            }
        } else {
            h = (sign << 15) | (newExp << 10) | (mantissa >>> 13);
        }
    }

    return h;
}

// ============================================================
// 恒等转换函数
// ============================================================

const identityConverter: DataConverter = (src: ArrayBuffer, _numel: number) => src;

// ============================================================
// 类型专用的转换函数
// ============================================================

// --- uint8: 扩展为 u32 ---
const uint8UploadConverter: DataConverter = (src: ArrayBuffer, numel: number) => {
    const srcArray = new Uint8Array(src, 0, numel);
    const dstBuffer = new ArrayBuffer(numel * 4);
    const dstView = new Uint32Array(dstBuffer);
    for (let i = 0; i < numel; i++) {
        dstView[i] = srcArray[i];
    }
    return dstBuffer;
};

const uint8DownloadConverter: DataConverter = (src: ArrayBuffer, numel: number) => {
    const srcView = new Uint32Array(src);
    const dstBuffer = new ArrayBuffer(numel);
    const dstArray = new Uint8Array(dstBuffer);
    for (let i = 0; i < numel; i++) {
        dstArray[i] = srcView[i];
    }
    return dstBuffer;
};

// --- int8: 扩展为 i32 ---
const int8UploadConverter: DataConverter = (src: ArrayBuffer, numel: number) => {
    const srcArray = new Int8Array(src, 0, numel);
    const dstBuffer = new ArrayBuffer(numel * 4);
    const dstView = new Int32Array(dstBuffer);
    for (let i = 0; i < numel; i++) {
        dstView[i] = srcArray[i];
    }
    return dstBuffer;
};

const int8DownloadConverter: DataConverter = (src: ArrayBuffer, numel: number) => {
    const srcView = new Int32Array(src);
    const dstBuffer = new ArrayBuffer(numel);
    const dstArray = new Int8Array(dstBuffer);
    for (let i = 0; i < numel; i++) {
        dstArray[i] = srcView[i];
    }
    return dstBuffer;
};

// --- uint16: 扩展为 u32 ---
const uint16UploadConverter: DataConverter = (src: ArrayBuffer, numel: number) => {
    const srcArray = new Uint16Array(src, 0, numel);
    const dstBuffer = new ArrayBuffer(numel * 4);
    const dstView = new Uint32Array(dstBuffer);
    for (let i = 0; i < numel; i++) {
        dstView[i] = srcArray[i];
    }
    return dstBuffer;
};

const uint16DownloadConverter: DataConverter = (src: ArrayBuffer, numel: number) => {
    const srcView = new Uint32Array(src);
    const dstBuffer = new ArrayBuffer(numel * 2);
    const dstArray = new Uint16Array(dstBuffer);
    for (let i = 0; i < numel; i++) {
        dstArray[i] = srcView[i];
    }
    return dstBuffer;
};

// --- int16: 扩展为 i32 ---
const int16UploadConverter: DataConverter = (src: ArrayBuffer, numel: number) => {
    const srcArray = new Int16Array(src, 0, numel);
    const dstBuffer = new ArrayBuffer(numel * 4);
    const dstView = new Int32Array(dstBuffer);
    for (let i = 0; i < numel; i++) {
        dstView[i] = srcArray[i];
    }
    return dstBuffer;
};

const int16DownloadConverter: DataConverter = (src: ArrayBuffer, numel: number) => {
    const srcView = new Int32Array(src);
    const dstBuffer = new ArrayBuffer(numel * 2);
    const dstArray = new Int16Array(dstBuffer);
    for (let i = 0; i < numel; i++) {
        dstArray[i] = srcView[i];
    }
    return dstBuffer;
};

// --- bool: 扩展为 u32 ---
const boolUploadConverter: DataConverter = (src: ArrayBuffer, numel: number) => {
    const srcArray = new Uint8Array(src, 0, numel);
    const dstBuffer = new ArrayBuffer(numel * 4);
    const dstView = new Uint32Array(dstBuffer);
    for (let i = 0; i < numel; i++) {
        dstView[i] = srcArray[i] !== 0 ? 1 : 0;
    }
    return dstBuffer;
};

const boolDownloadConverter: DataConverter = (src: ArrayBuffer, numel: number) => {
    const srcView = new Uint32Array(src);
    const dstBuffer = new ArrayBuffer(numel);
    const dstArray = new Uint8Array(dstBuffer);
    for (let i = 0; i < numel; i++) {
        dstArray[i] = srcView[i] !== 0 ? 1 : 0;
    }
    return dstBuffer;
};

// --- float64: 降级为 f32 ---
const float64UploadConverter: DataConverter = (src: ArrayBuffer, numel: number) => {
    const srcArray = new Float64Array(src, 0, numel);
    const dstBuffer = new ArrayBuffer(numel * 4);
    const dstView = new Float32Array(dstBuffer);
    for (let i = 0; i < numel; i++) {
        dstView[i] = srcArray[i];
    }
    return dstBuffer;
};

const float64DownloadConverter: DataConverter = (src: ArrayBuffer, numel: number) => {
    const srcView = new Float32Array(src);
    const dstBuffer = new ArrayBuffer(numel * 8);
    const dstView = new Float64Array(dstBuffer);
    for (let i = 0; i < numel; i++) {
        dstView[i] = srcView[i];
    }
    return dstBuffer;
};

// --- int64: 降级为 i32 ---
const int64UploadConverter: DataConverter = (src: ArrayBuffer, numel: number) => {
    const srcArray = new BigInt64Array(src, 0, numel);
    const dstBuffer = new ArrayBuffer(numel * 4);
    const dstView = new Int32Array(dstBuffer);
    for (let i = 0; i < numel; i++) {
        dstView[i] = Number(srcArray[i]);
    }
    return dstBuffer;
};

const int64DownloadConverter: DataConverter = (src: ArrayBuffer, numel: number) => {
    const srcView = new Int32Array(src);
    const dstBuffer = new ArrayBuffer(numel * 8);
    const dstView = new BigInt64Array(dstBuffer);
    for (let i = 0; i < numel; i++) {
        dstView[i] = BigInt(srcView[i]);
    }
    return dstBuffer;
};

// --- uint64: 降级为 u32 ---
const uint64UploadConverter: DataConverter = (src: ArrayBuffer, numel: number) => {
    const srcArray = new BigUint64Array(src, 0, numel);
    const dstBuffer = new ArrayBuffer(numel * 4);
    const dstView = new Uint32Array(dstBuffer);
    for (let i = 0; i < numel; i++) {
        dstView[i] = Number(srcArray[i]);
    }
    return dstBuffer;
};

const uint64DownloadConverter: DataConverter = (src: ArrayBuffer, numel: number) => {
    const srcView = new Uint32Array(src);
    const dstBuffer = new ArrayBuffer(numel * 8);
    const dstView = new BigUint64Array(dstBuffer);
    for (let i = 0; i < numel; i++) {
        dstView[i] = BigInt(srcView[i]);
    }
    return dstBuffer;
};

// --- complex128: Float64 降级为 Float32 ---
const complex128UploadConverter: DataConverter = (src: ArrayBuffer, numel: number) => {
    // numel 是复数元素数量，实际 float 数量 = numel * 2
    const srcArray = new Float64Array(src, 0, numel * 2);
    const dstBuffer = new ArrayBuffer(numel * 8); // 每个复数 = 2 * f32 = 8 bytes
    const dstView = new Float32Array(dstBuffer);
    for (let i = 0; i < numel * 2; i++) {
        dstView[i] = srcArray[i];
    }
    return dstBuffer;
};

const complex128DownloadConverter: DataConverter = (src: ArrayBuffer, numel: number) => {
    // GPU 返回的是 Float32Array，每个复数 = 2 个 f32
    const srcView = new Float32Array(src);
    const dstBuffer = new ArrayBuffer(numel * 16); // 每个复数 = 2 * f64 = 16 bytes
    const dstView = new Float64Array(dstBuffer);
    for (let i = 0; i < numel * 2; i++) {
        dstView[i] = srcView[i];
    }
    return dstBuffer;
};

// --- float16 (设备支持 f16): f32 values <-> f16 bits ---
const float16NativeUploadConverter: DataConverter = (src: ArrayBuffer, numel: number) => {
    const srcArray = new Float32Array(src, 0, numel);
    const rawBytes = numel * 2;
    const alignedBytes = Math.ceil(rawBytes / 4) * 4; // 4-byte 对齐
    const dstBuffer = new ArrayBuffer(alignedBytes);
    const dstView = new Uint16Array(dstBuffer);
    for (let i = 0; i < numel; i++) {
        dstView[i] = float32ToFloat16(srcArray[i]);
    }
    return dstBuffer;
};

const float16NativeDownloadConverter: DataConverter = (src: ArrayBuffer, numel: number) => {
    const srcView = new Uint16Array(src);
    const dstBuffer = new ArrayBuffer(numel * 4);
    const dstArray = new Float32Array(dstBuffer);
    for (let i = 0; i < numel; i++) {
        dstArray[i] = float16ToFloat32(srcView[i]);
    }
    return dstBuffer;
};

// --- float16 (设备不支持 f16): 使用 f32，无需转换 ---
// 直接使用 identityConverter

// ============================================================
// DTypeResolver 实现
// ============================================================

class DTypeResolverImpl implements IDTypeResolver {
    private readonly descriptors: Map<DType, PhysicalStorageDescriptor>;
    readonly supportsNativeF16: boolean;
    readonly float16StoragePrecision: 'f16' | 'f32';

    constructor(supportsF16: boolean) {
        this.supportsNativeF16 = supportsF16;
        this.float16StoragePrecision = supportsF16 ? 'f16' : 'f32';
        this.descriptors = this.buildDescriptors(supportsF16);
    }

    private buildDescriptors(supportsF16: boolean): Map<DType, PhysicalStorageDescriptor> {
        const descriptors = new Map<DType, PhysicalStorageDescriptor>();

        // === bool ===
        descriptors.set('bool', {
            logicalDType: 'bool',
            wgslStorageType: 'u32',
            wgslComputeType: 'bool', // 计算时用 bool，存储时用 u32
            gpuBytesPerElement: 4,
            jsTypedArrayCtor: Uint8Array,
            needsUploadConversion: true,
            needsDownloadConversion: true,
            uploadConverter: boolUploadConverter,
            downloadConverter: boolDownloadConverter,
        });

        // === uint8 ===
        descriptors.set('uint8', {
            logicalDType: 'uint8',
            wgslStorageType: 'u32',
            wgslComputeType: 'u32',
            gpuBytesPerElement: 4,
            jsTypedArrayCtor: Uint8Array,
            needsUploadConversion: true,
            needsDownloadConversion: true,
            uploadConverter: uint8UploadConverter,
            downloadConverter: uint8DownloadConverter,
        });

        // === int8 ===
        descriptors.set('int8', {
            logicalDType: 'int8',
            wgslStorageType: 'i32',
            wgslComputeType: 'i32',
            gpuBytesPerElement: 4,
            jsTypedArrayCtor: Int8Array,
            needsUploadConversion: true,
            needsDownloadConversion: true,
            uploadConverter: int8UploadConverter,
            downloadConverter: int8DownloadConverter,
        });

        // === uint16 ===
        descriptors.set('uint16', {
            logicalDType: 'uint16',
            wgslStorageType: 'u32',
            wgslComputeType: 'u32',
            gpuBytesPerElement: 4,
            jsTypedArrayCtor: Uint16Array,
            needsUploadConversion: true,
            needsDownloadConversion: true,
            uploadConverter: uint16UploadConverter,
            downloadConverter: uint16DownloadConverter,
        });

        // === int16 ===
        descriptors.set('int16', {
            logicalDType: 'int16',
            wgslStorageType: 'i32',
            wgslComputeType: 'i32',
            gpuBytesPerElement: 4,
            jsTypedArrayCtor: Int16Array,
            needsUploadConversion: true,
            needsDownloadConversion: true,
            uploadConverter: int16UploadConverter,
            downloadConverter: int16DownloadConverter,
        });

        // === uint32 (原生支持) ===
        descriptors.set('uint32', {
            logicalDType: 'uint32',
            wgslStorageType: 'u32',
            wgslComputeType: 'u32',
            gpuBytesPerElement: 4,
            jsTypedArrayCtor: Uint32Array,
            needsUploadConversion: false,
            needsDownloadConversion: false,
            uploadConverter: identityConverter,
            downloadConverter: identityConverter,
        });

        // === int32 (原生支持) ===
        descriptors.set('int32', {
            logicalDType: 'int32',
            wgslStorageType: 'i32',
            wgslComputeType: 'i32',
            gpuBytesPerElement: 4,
            jsTypedArrayCtor: Int32Array,
            needsUploadConversion: false,
            needsDownloadConversion: false,
            uploadConverter: identityConverter,
            downloadConverter: identityConverter,
        });

        // === float32 (原生支持) ===
        descriptors.set('float32', {
            logicalDType: 'float32',
            wgslStorageType: 'f32',
            wgslComputeType: 'f32',
            gpuBytesPerElement: 4,
            jsTypedArrayCtor: Float32Array,
            needsUploadConversion: false,
            needsDownloadConversion: false,
            uploadConverter: identityConverter,
            downloadConverter: identityConverter,
        });

        // === float16 (根据设备能力) ===
        if (supportsF16) {
            descriptors.set('float16', {
                logicalDType: 'float16',
                wgslStorageType: 'f16',
                wgslComputeType: 'f32', // 即使存储用 f16，计算通常用 f32 保精度
                gpuBytesPerElement: 2,
                jsTypedArrayCtor: Float32Array, // 用户始终拿到 Float32Array
                needsUploadConversion: true,
                needsDownloadConversion: true,
                uploadConverter: float16NativeUploadConverter,
                downloadConverter: float16NativeDownloadConverter,
            });
        } else {
            descriptors.set('float16', {
                logicalDType: 'float16',
                wgslStorageType: 'f32',
                wgslComputeType: 'f32',
                gpuBytesPerElement: 4,
                jsTypedArrayCtor: Float32Array,
                needsUploadConversion: false,
                needsDownloadConversion: false,
                uploadConverter: identityConverter,
                downloadConverter: identityConverter,
            });
        }

        // === float64 (降级为 f32) ===
        descriptors.set('float64', {
            logicalDType: 'float64',
            wgslStorageType: 'f32',
            wgslComputeType: 'f32',
            gpuBytesPerElement: 4,
            jsTypedArrayCtor: Float64Array,
            needsUploadConversion: true,
            needsDownloadConversion: true,
            uploadConverter: float64UploadConverter,
            downloadConverter: float64DownloadConverter,
        });

        // === int64 (降级为 i32) ===
        descriptors.set('int64', {
            logicalDType: 'int64',
            wgslStorageType: 'i32',
            wgslComputeType: 'i32',
            gpuBytesPerElement: 4,
            jsTypedArrayCtor: BigInt64Array,
            needsUploadConversion: true,
            needsDownloadConversion: true,
            uploadConverter: int64UploadConverter,
            downloadConverter: int64DownloadConverter,
        });

        // === uint64 (降级为 u32) ===
        descriptors.set('uint64', {
            logicalDType: 'uint64',
            wgslStorageType: 'u32',
            wgslComputeType: 'u32',
            gpuBytesPerElement: 4,
            jsTypedArrayCtor: BigUint64Array,
            needsUploadConversion: true,
            needsDownloadConversion: true,
            uploadConverter: uint64UploadConverter,
            downloadConverter: uint64DownloadConverter,
        });

        // === complex64 (使用 vec2<f32>) ===
        descriptors.set('complex64', {
            logicalDType: 'complex64',
            wgslStorageType: 'vec2<f32>',
            wgslComputeType: 'vec2<f32>',
            gpuBytesPerElement: 8, // vec2<f32> = 2 * 4 bytes
            jsTypedArrayCtor: Float32Array,
            needsUploadConversion: false, // Float32Array 布局兼容 vec2<f32>
            needsDownloadConversion: false,
            uploadConverter: identityConverter,
            downloadConverter: identityConverter,
        });

        // === complex128 (降级为 vec2<f32>) ===
        descriptors.set('complex128', {
            logicalDType: 'complex128',
            wgslStorageType: 'vec2<f32>',
            wgslComputeType: 'vec2<f32>',
            gpuBytesPerElement: 8, // 降级后为 vec2<f32>
            jsTypedArrayCtor: Float64Array,
            needsUploadConversion: true,
            needsDownloadConversion: true,
            uploadConverter: complex128UploadConverter,
            downloadConverter: complex128DownloadConverter,
        });

        return descriptors;
    }

    getDescriptor(dtype: DType): PhysicalStorageDescriptor {
        const descriptor = this.descriptors.get(dtype);
        if (!descriptor) {
            throw new Error(`Unsupported dtype: ${dtype}`);
        }
        return descriptor;
    }

    calculateStorageBytes(dtype: DType, numel: number): number {
        const descriptor = this.getDescriptor(dtype);
        const rawBytes = numel * descriptor.gpuBytesPerElement;
        // WebGPU 要求 buffer 大小必须是 4 的倍数
        return Math.ceil(rawBytes / 4) * 4;
    }
}

// ============================================================
// 工厂函数
// ============================================================

/**
 * 构建 DType 解析器
 * 
 * @param supportsF16 设备是否支持 shader-f16 扩展
 * @returns DTypeResolver 实例
 */
export function buildDTypeResolver(supportsF16: boolean): IDTypeResolver {
    return new DTypeResolverImpl(supportsF16);
}

// ============================================================
// 全局 Resolver 实例管理
// ============================================================

let globalResolver: IDTypeResolver | null = null;

/**
 * 初始化全局 DType 解析器
 * 
 * 应在 WebGPUDeviceManager.init() 之后调用一次
 */
export function initGlobalDTypeResolver(supportsF16: boolean): void {
    globalResolver = buildDTypeResolver(supportsF16);
}

/**
 * 获取全局 DType 解析器
 * 
 * @throws 如果尚未初始化
 */
export function getGlobalDTypeResolver(): IDTypeResolver {
    if (!globalResolver) {
        throw new Error('DTypeResolver has not been initialized. Call initGlobalDTypeResolver() first.');
    }
    return globalResolver;
}

/**
 * 重置全局 DType 解析器（用于测试）
 */
export function resetGlobalDTypeResolver(): void {
    globalResolver = null;
}
