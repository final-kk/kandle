/**
 * WebGPU DType Utilities
 * 
 * 此文件包含 WGSL shader 相关的类型常量和工具函数。
 * 注意：所有与物理存储相关的逻辑已迁移到 DTypeResolver.ts
 */

import { DType } from "@kandle/types";
import { WgslDType } from "../types";

// ============================================================
// WGSL Shader Constants
// ============================================================

/**
 * WGSL constants for reduction operations
 * Using hexadecimal float literals for FLT_MAX (WGSL standard)
 */
export const WGSL_CONSTANTS = {
    /** FLT_MAX: Maximum representable f32 value */
    FLT_MAX: '0x1.fffffep+127',
    /** -FLT_MAX: Negative maximum (for max reduction initialization) */
    NEG_FLT_MAX: '-0x1.fffffep+127',
    /** INT_MAX: Maximum i32 value */
    INT_MAX: '2147483647',
    /** INT_MIN: Minimum i32 value */
    INT_MIN: '-2147483648',
    /** UINT_MAX: Maximum u32 value (2^32 - 1) */
    UINT_MAX: '4294967295',
    /** PI: Mathematical constant π */
    PI: '3.14159265359',
} as const;

// ============================================================
// WGSL Compute Type Mapping
// ============================================================

/**
 * 计算时使用的 WGSL 类型映射表
 * 
 * 作用：决定了在 Shader 中 let a = ...; let b = ...; 时 a 和 b 的类型。
 * 
 * 逻辑：
 * 1. 所有的 float (含 f64) -> f32
 * 2. 所有的 int (含 i8, i16, i64) -> i32
 * 3. 所有的 uint (含 u8, u16, u64) -> u32
 * 4. bool -> bool (但在存储层是 u32，加载时需要 cast)
 * 5. complex -> vec2<f32>
 */
export const WEBGPU_COMPUTE_DTYPE_MAP: Record<DType, WgslDType> = {
    // Boolean
    "bool": "bool", // 注意：计算时用 bool，存储时用 u32

    // Integers - 统一提升到 32位 寄存器计算
    "int8": "i32",
    "int16": "i32",
    "int32": "i32",
    "int64": "i32", // 降级

    // Unsigned Integers
    "uint8": "u32",
    "uint16": "u32",
    "uint32": "u32",
    "uint64": "u32", // 降级

    // Floats - 统一用 f32 计算
    "float16": "f32", // 提升到 f32 计算，防止精度溢出
    "float32": "f32",
    "float64": "f32", // 降级

    // Complex
    "complex64": "vec2<f32>",
    "complex128": "vec2<f32>"
};

// ============================================================
// Utility Functions
// ============================================================

/**
 * 获取逻辑类型对应的 WGSL 计算类型
 * 
 * @param dtype 逻辑类型
 * @returns WGSL 计算类型 (f32, i32, u32, bool, vec2<f32>)
 */
export function getComputeType(dtype: DType): WgslDType {
    const type = WEBGPU_COMPUTE_DTYPE_MAP[dtype];
    if (!type) {
        throw new Error(`Unsupported compute dtype: ${dtype}`);
    }
    return type;
}

/**
 * 生成 WGSL 类型转换代码片段
 * 
 * @param valueVar 变量名
 * @param fromType 当前变量的物理存储 WGSL 类型
 * @param toType 目标计算 WGSL 类型
 * @returns WGSL 表达式字符串
 */
export function generateCastSnippet(valueVar: string, fromType: WgslDType, toType: WgslDType): string {
    if (fromType === toType) {
        return valueVar;
    }

    // 特殊处理：Bool 类型的转换
    // 场景 1: Storage(u32) -> Compute(bool)
    if (fromType === 'u32' && toType === 'bool') {
        return `(${valueVar} != 0u)`;
    }
    // 场景 2: Compute(bool) -> Storage(u32) (通常用于 Output)
    if (fromType === 'bool' && toType === 'u32') {
        return `select(0u, 1u, ${valueVar})`;
    }

    // 特殊处理：向量 (Complex)
    // 如果之后涉及到 scalar -> complex 的广播，这里需要处理 (e.g. f32 -> vec2<f32>)
    if (toType === 'vec2<f32>' && fromType === 'f32') {
        return `vec2<f32>(${valueVar}, 0.0)`;
    }

    // Special case: Complex (vector) -> Scalar (f32)
    // Used for abs/real operations where we perform the op in complex domain (returning vec2 real)
    // but store into float buffer. We just take the real component.
    if (fromType === 'vec2<f32>' && toType === 'f32') {
        return `${valueVar}.x`;
    }

    // 标准标量转换：T(value)
    // 例如：f32(u32_val), i32(u8_simulated_u32)
    return `${toType}(${valueVar})`;
}

// ============================================================
// Float16 Conversion Utilities (Re-exported from DTypeResolver)
// ============================================================

// Note: Float16 conversion functions are now in DTypeResolver.ts
// Export them from here for backwards compatibility if needed
export { float16ToFloat32, float32ToFloat16 } from './DTypeResolver';
