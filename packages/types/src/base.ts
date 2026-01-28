/**
 * DType Mapping for strict type inference.
 * Defines how logical types map to physical storage in JS.
 */
export interface DataTypeMap {
    bool: Uint8Array;
    int8: Int8Array;
    uint8: Uint8Array;
    uint16: Uint16Array;
    int16: Int16Array;
    uint32: Uint32Array;
    float16: Float32Array; // decoded to float32 for user consumption (JS has no native Float16Array)
    int32: Int32Array;
    float32: Float32Array;
    int64: BigInt64Array;
    uint64: BigUint64Array;
    float64: Float64Array;
    complex64: Float32Array;   // stored as [real, imag]
    complex128: Float64Array;  // stored as [real, imag]
}

/**
 * The supported data types (e.g., 'float32', 'int32')
 */
export type DType = keyof DataTypeMap;

/**
 * @description some case where no dtype is specified, and should be a float, then use float32  
 */
export const DEFAULT_FLOAT_DTYPE: DType = 'float32';

/**
 * @description some case where no dtype is specified, and should be an integer, then use int32
 */
export const DEFAULT_INT_DTYPE: DType = 'int32';

export enum DTypeCategory {
    Bool = 0,
    Integral = 1,
    Floating = 2,
    Complex = 3,
}

export interface DTypeInfo {
    category: DTypeCategory;
    bitWidth: 8 | 16 | 32 | 64 | 128;
    isSigned: boolean;
}

export const TYPE_REGISTRY: Record<DType, DTypeInfo> = {
    bool: { category: DTypeCategory.Bool, bitWidth: 8, isSigned: false },
    uint8: { category: DTypeCategory.Integral, bitWidth: 8, isSigned: false },
    int8: { category: DTypeCategory.Integral, bitWidth: 8, isSigned: true },
    int16: { category: DTypeCategory.Integral, bitWidth: 16, isSigned: true },
    uint16: { category: DTypeCategory.Integral, bitWidth: 16, isSigned: false },
    float16: { category: DTypeCategory.Floating, bitWidth: 16, isSigned: true },
    uint32: { category: DTypeCategory.Integral, bitWidth: 32, isSigned: false },
    int32: { category: DTypeCategory.Integral, bitWidth: 32, isSigned: true },
    float32: { category: DTypeCategory.Floating, bitWidth: 32, isSigned: true },
    uint64: { category: DTypeCategory.Integral, bitWidth: 64, isSigned: false },
    int64: { category: DTypeCategory.Integral, bitWidth: 64, isSigned: true },
    float64: { category: DTypeCategory.Floating, bitWidth: 64, isSigned: true },
    complex64: { category: DTypeCategory.Complex, bitWidth: 64, isSigned: true },
    complex128: { category: DTypeCategory.Complex, bitWidth: 128, isSigned: true },
};

/**
 * Union of all supported underlying data containers
 */
export type TensorData = DataTypeMap[DType];

export type ScalarDataType = number | bigint | boolean;

export type RecursiveArray<T = ScalarDataType> = T[] | RecursiveArray<T>[];

export type StructureDataType =
    | TensorData
    | RecursiveArray<ScalarDataType>

export type TensorDataLike =
    | StructureDataType
    | ScalarDataType;

/**
 * Standard Shape definition
 */
export type Shape = readonly number[];

export type ShapeLike = Shape | number[];

export interface SliceParam {
    start: number;
    end: number;
    step: number;
}

export enum DeviceNameEnum {
    JS = "js",
    Node = "node",
    WebGPU = "webgpu",
    WASM = "wasm",
}

export enum MemoryFormat {
    Contiguous = "contiguous",
    ChannelsLast = "channels_last",
    ChannelsLast3d = "channels_last_3d",
    Preserve = "preserve",
}

/**
 * Type Promotion Kind - 控制类型推导策略
 * 
 * 对应 PyTorch 的 ELEMENTWISE_TYPE_PROMOTION_KIND:
 * - match_input: 标准类型提升，结果类型与输入相同 (add, mul, ...)
 * - always_bool: 恒返回 bool (eq, lt, isnan, ...)
 * - always_int64: 恒返回 int64 (argmax, argmin, ...)
 * - always_float: 整数输入时提升到浮点 (div, mean, ...)
 * - integer_promotion: 整数提升到 int64 防溢出 (sum, prod, ...)
 */
export type TypePromotionKind =
    | 'match_input'
    | 'always_bool'
    | 'always_int64'
    | 'always_float'
    | 'integer_promotion';

// ============================================================================
// Complex Type System
// ============================================================================

/**
 * 复数接口 - 用于用户交互层
 * 
 * 设计决策:
 * 1. 显式构造 - 用户必须使用 complex() 函数创建复数
 * 2. 不支持嵌套数组 - 避免 [[1,2]] 形式的歧义
 * 3. 内部存储 - Float32Array/Float64Array 交错 [r,i,r,i,...]
 * 4. __complex__ 标记 - 防止普通对象误识别
 * 
 * 注意: 仅用于构造函数输入和 .item() 返回
 * 内部存储始终使用 Float32Array/Float64Array
 */
export interface Complex {
    readonly re: number;
    readonly im: number;
    readonly __complex__: true;
}