import { DEFAULT_FLOAT_DTYPE, DType, ITensorHandle, TYPE_REGISTRY, DTypeCategory, DTypeInfo, getOpEntry, type OpEntry, type OpName, TypePromotionKind } from "@kandle/types";

export function getTypedArrayCtor(dtype: DType): any {
    switch (dtype) {
        case "float32":
            return Float32Array;
        case "float64":
            return Float64Array;
        case "int32":
            return Int32Array;
        case "int16":
            return Int16Array;
        case "int8":
            return Int8Array;
        case "uint32":
            return Uint32Array;
        case "uint16":
            return Uint16Array;
        case "uint8":
            return Uint8Array;
        case "int64":
            return BigInt64Array;
        case "uint64":
            return BigUint64Array;
        case "bool":
            return Uint8Array;
        case "float16":
            return Float32Array;  // decoded to float32 (JS has no native Float16Array)
        case "complex64":
            return Float32Array;  // [real, imag] pairs
        case "complex128":
            return Float64Array;  // [real, imag] pairs
        default:
            throw new Error(`Unsupported DType: ${dtype}`);
    }
}

export function getDTypeFromTypedArray(data: any): DType {
    if (data instanceof Float32Array) return "float32";
    if (data instanceof Float64Array) return "float64";
    if (data instanceof Int32Array) return "int32";
    if (data instanceof Int16Array) return "int16";
    if (data instanceof Int8Array) return "int8";
    if (data instanceof Uint32Array) return "uint32";
    if (data instanceof Uint16Array) return "uint16";
    if (data instanceof Uint8Array) return "uint8";
    if (data instanceof BigInt64Array) return "int64";
    if (data instanceof BigUint64Array) return "uint64";
    return "float32";
}

export function isTypedArray(data: any): boolean {
    return ArrayBuffer.isView(data) && !(data instanceof DataView);
}


const FLOAT_TYPES = new Set<DType>(['float16', 'float32', 'float64']);
const COMPLEX_TYPES = new Set<DType>(['complex64', 'complex128']);
// 注意: BigInt64/Uint64 虽然是整数，但在 JS 处理上稍有不同，这里归类为 Integer
const INT_TYPES = new Set<DType>([
    'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64'
]);

export const isFloatingPoint = (dt: DType): boolean => FLOAT_TYPES.has(dt);
/** 检查 dtype 是否是复数类型 */
export const isComplexDtype = (dt: DType): boolean => COMPLEX_TYPES.has(dt);
export const isInteger = (dt: DType): boolean => INT_TYPES.has(dt);
export const isBool = (dt: DType): boolean => dt === 'bool';

// ============================================================================
// Helper: Get OpEntry from OpRegistry (v5)
// ============================================================================

/**
 * Get the operator entry from the registry.
 * Returns undefined if not found.
 */
export function getOpDef(opName: OpName): OpEntry | undefined {
    return getOpEntry(opName);
}

/**
 * Determine if an operator is a "division-like" operation
 * (requires floating point output for integer inputs)
 */
function isDivisionOp(opName: string): boolean {
    return opName === 'div';
}

/**
 * Determine if an operator is a "transcendental" operation
 * (sin, cos, exp, log, etc. - requires floating point)
 */
function isTranscendentalOp(opName: string): boolean {
    const transcendentals = [
        'exp', 'log', 'log2', 'log10', 'log1p',
        'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
        'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
        'sqrt', 'rsqrt', 'erf', 'erfc', 'erfinv',
    ];
    return transcendentals.includes(opName);
}

/**
 * Determine if an operator is a "comparison" operation
 */
function isComparisonOp(opName: string): boolean {
    const comparisons = [
        'eq', 'ne', 'lt', 'le', 'gt', 'ge',
        'equal', 'notEqual', 'less', 'lessEqual', 'greater', 'greaterEqual',
    ];
    return comparisons.includes(opName);
}

/**
 * Determine if an operator is a "bitwise" operation
 */
function isBitwiseOp(opName: string): boolean {
    const bitwiseOps = [
        'bitwiseAnd', 'bitwiseOr', 'bitwiseXor', 'bitwiseNot',
        'leftShift', 'rightShift',
    ];
    return bitwiseOps.includes(opName);
}

/**
 * Determine if an operator is "arithmetic" (add, sub, mul, etc.)
 */
function isArithmeticOp(opName: string): boolean {
    const arithmeticOps = [
        'add', 'sub', 'mul', 'mod', 'maximum', 'minimum',
        'floorDivide', 'remainder', 'pow',
        'abs', 'real', 'imag', 'angle', 'arg'
    ];
    return arithmeticOps.includes(opName);
}

/**
 * Determine if an operator converts complex input to real output
 */
function isComplexToRealOp(opName: string): boolean {
    return ['abs', 'real', 'imag', 'angle', 'arg'].includes(opName);
}

/**
 * @description compute Tensor + Scalar result DType
 */
/**
 * 计算 Tensor + Scalar 的结果类型
 * 完全对标 PyTorch Native Type Promotion
 */
export function computeScalarResultDType(
    t: ITensorHandle,
    scalar: number | bigint,
    opName: OpName = "add"
): DType {

    const tType = t.dtype;

    // 1. 位运算检查: 必须是整数
    if (isBitwiseOp(opName)) {
        if (!isInteger(tType) && !isBool(tType)) {
            throw new Error(`Bitwise operations only support integer tensors, got ${tType}`);
        }
        if (typeof scalar === 'number' && !Number.isInteger(scalar)) {
            throw new Error("Bitwise operations requires integer scalar");
        }
        // 位运算通常维持原类型，或者提升到能够容纳标量的整数类型
        return tType;
    }

    // 2. 除法特例: 整数除法必须转浮点
    if (isDivisionOp(opName)) {
        if (isFloatingPoint(tType) || isComplexDtype(tType)) return tType;
        return DEFAULT_FLOAT_DTYPE;
    }

    // Complex64 + 1 => Complex64
    if (isComplexDtype(tType)) {
        return tType;
    }

    // Float Tensor + Any Real Scalar => Float
    // Float16 + 100 => Float16
    // Float32 + 1.5 => Float32
    if (isFloatingPoint(tType)) {
        return tType;
    }

    // BigInt Scalar + Tensor
    if (typeof scalar === 'bigint') {
        // If Tensor is Integer (Int8...Int64), must promote to Int64 to accommodate BigInt
        // Int8 + 10n => Int64
        if (isInteger(tType) || isBool(tType)) {
            return 'int64'; // as torch does
        }

        return tType;
    }

    const isScalarInt = Number.isInteger(scalar);

    if (isScalarInt) {

        if (isBool(tType)) {
            // special case: Bool Tensor + 1 => Int64   (as torch does)
            return 'int64';
        }

        // Int8 + 5 => Int8
        // Uint16 + 10 => Uint16
        return tType;
    } else {
        // Integer/Bool Tensor + Float Scalar => Float32
        // Int8 + 1.5 => Float32
        return DEFAULT_FLOAT_DTYPE;
    }
}

// 1. 定义类别优先级：Complex > Float > Int > Bool


export function getDTypeInfo(dtype: DType): DTypeInfo {
    const info = TYPE_REGISTRY[dtype];
    if (!info) throw new Error(`Unknown dtype: ${dtype}`);
    return info;
}

/**
 * 决定两个类型的提升结果
 * 对应 PyTorch c10::promoteTypes
 */
export function promoteTypes(typeA: DType, typeB: DType): DType {
    if (typeA === typeB) return typeA;

    const infoA = getDTypeInfo(typeA);
    const infoB = getDTypeInfo(typeB);

    // 规则 1: 类别不同，高等级获胜 (Complex > Float > Int > Bool)
    if (infoA.category !== infoB.category) {
        return infoA.category > infoB.category ? typeA : typeB;
    }

    // 规则 2: 类别相同，位宽不同，大位宽获胜
    if (infoA.bitWidth !== infoB.bitWidth) {
        return infoA.bitWidth > infoB.bitWidth ? typeA : typeB;
    }

    // 规则 3: 类别、位宽都相同 (主要是 int vs uint)，有符号获胜
    // 例如: int32 vs uint32 -> int32 (PyTorch偏向signed，虽然这可能导致uint溢出，但在同宽下由signed主导)
    if (infoA.isSigned !== infoB.isSigned) {
        return infoA.isSigned ? typeA : typeB;
    }

    return typeA;
}

/**
 * 处理多元情况 (e.g. stack, cat, where, or a + b + c)
 * 输入是一个 dtype 列表，返回最终共同的 dtype
 */
export function resolveCommonDType(dtypes: DType[]): DType {
    if (dtypes.length === 0) return DEFAULT_FLOAT_DTYPE;

    let currentType = dtypes[0];

    for (let i = 1; i < dtypes.length; i++) {
        currentType = promoteTypes(currentType, dtypes[i]);
    }

    return currentType;
}

/**
 * 决定在 Kernel 计算时使用的 Accumulation Type
 * 对应 PyTorch native::toAccType
 */
export function resolveComputationType(dtype: DType, opName: OpName): DType {

    const info = getDTypeInfo(dtype);
    const op = getOpDef(opName);

    // --- 规则 1: 浮点数总是尽量保持精度 ---
    // Float16/BFloat16 -> Float32 计算 (为了数值稳定性)
    // 无论是什么算子 (除了 copy/move)，Float16 都不适合直接做累加
    if (info.category === DTypeCategory.Floating) {
        if (info.bitWidth < 32) {
            return DEFAULT_FLOAT_DTYPE;
        }
        return dtype; // float32, float64 保持不变
    }

    // --- 规则 2: 复数 ---
    // complex64 -> complex64 (实部/虚部用 float32 存储，精度已足够)
    // complex128 -> complex128 (实部/虚部用 float64 存储)
    if (info.category === DTypeCategory.Complex) {
        return dtype;
    }

    // --- 规则 3: 整数及布尔值的特殊处理 ---
    if (info.category === DTypeCategory.Integral || info.category === DTypeCategory.Bool) {

        // Case A: 必须转浮点的算子 (Division, Transcendental, Mean)
        // mean(int) -> float, div(int, int) -> float, sin(int) -> float
        if (
            isDivisionOp(opName) ||
            isTranscendentalOp(opName) ||
            opName === 'mean' // 特例：mean 必须是 float
        ) {
            return DEFAULT_FLOAT_DTYPE;
        }

        // Case B: 归约求和 (Sum) -> 转 Int64
        // sum(int8) -> int64 (为了防止溢出)
        if (opName === 'sum') {
            return 'int64';
        }

        // Case C: 位运算 -> 必须保持原样 (甚至不应该做提升，但假设这里传入的是 Int)
        if (isBitwiseOp(opName)) {
            // 可以在这里加个断言：if (dtype is float) throw Error
            return dtype;
        }

        // Case D: 比较运算 (eq, lt, gt...) -> 保持原样进行比较
        // 输入类型决定比较逻辑，结果类型由 resolveResultType 决定 (always_bool)
        if (isComparisonOp(opName)) {
            return dtype;
        }

        // Case E: 普通算术 (Add, Mul, Sub, Max, Min) -> 保持原样
        // PyTorch 中: int32 + int32 = int32 (允许溢出)
        if (isArithmeticOp(opName) || opName === 'max' || opName === 'min') {
            if (info.category === DTypeCategory.Bool) {
                // Bool arithmetic must promote to integer (e.g. 1 + 1 = 2)
                // WGSL does not support bool + bool
                return 'int32';
            }
            return dtype;
        }
    }

    return dtype;

}

/**
 * Derive TypePromotionKind from OpEntry's dtype rule
 * 
 * This function can be used by handlers to extract the type promotion policy
 * from an OpEntry and pass it to TensorIterator.
 * 
 * @param dtypeRule - The dtype rule from OpEntry
 * @returns The type promotion kind
 */
export function deriveTypePromotionKindFromRule(
    dtypeRule: OpEntry['dtype']
): TypePromotionKind {
    if (dtypeRule.rule === 'fixed') {
        if (dtypeRule.dtype === 'bool') return 'always_bool';
        if (dtypeRule.dtype === 'int64') return 'always_int64';
    }

    if (dtypeRule.rule === 'promote') {
        if (dtypeRule.toCategory === 'Floating') return 'always_float';
        if (dtypeRule.toCategory === 'SignedIntegral') return 'integer_promotion';
    }

    return 'match_input';
}

/**
 * Resolve the result dtype based on the type promotion policy
 * 
 * @param commonDtype - The common dtype after type promotion of inputs
 * @param policy - The type promotion policy (directly passed, not looked up)
 * @param opName - Optional operation name for special cases
 * @returns The result dtype
 */
export function resolveResultTypeWithPolicy(
    commonDtype: DType,
    policy: TypePromotionKind,
    opName?: OpName
): DType {
    const info = getDTypeInfo(commonDtype);

    switch (policy) {
        case 'always_bool':
            return 'bool';

        case 'always_int64':
            return 'int64';

        case 'always_float':
            // Complex -> Float for abs/real/imag
            if (info.category === DTypeCategory.Complex && opName && isComplexToRealOp(opName)) {
                if (commonDtype === 'complex128') return 'float64';
                return 'float32';
            }

            // 如果已经是浮点或复数，保持原样(除非你想强制 float32)
            // 如果是整数/布尔，必须提升到默认浮点
            if (info.category === DTypeCategory.Integral || info.category === DTypeCategory.Bool) {
                return DEFAULT_FLOAT_DTYPE;
            }
            return commonDtype;

        case 'integer_promotion':
            // 专为 sum 设计
            // 整数 -> int64 (防溢出)
            // 浮点 -> 保持原样 (float16 sum 结果还是 float16，虽然计算用 float32)
            if (info.category === DTypeCategory.Integral || info.category === DTypeCategory.Bool) {
                return 'int64';
            }
            return commonDtype;

        case 'match_input':
        default:
            // Boolean arithmetic -> Int32
            // tests expect [2, 1, 1, 0] for bool add, so result must be numeric
            if (info.category === DTypeCategory.Bool && opName && isArithmeticOp(opName)) {
                return 'int32';
            }

            // Complex -> Float for abs/real/imag
            if (info.category === DTypeCategory.Complex && opName && isComplexToRealOp(opName)) {
                if (commonDtype === 'complex128') return 'float64';
                return 'float32';
            }

            // 大部分算子 (add, mul, sub...)
            // 输入是什么，输出就是什么
            return commonDtype;
    }
}

/**
 * Resolve the result dtype for an operation (backward compatible version)
 * 
 * This function looks up the OpEntry by opName to determine the type promotion policy.
 * For better reliability, prefer using resolveResultTypeWithPolicy with an explicitly
 * derived policy from OpEntry.dtype.
 * 
 * @param commonDtype - The common dtype after type promotion of inputs
 * @param opName - The operation name to look up in OpRegistry
 * @returns The result dtype
 */
export function resolveResultType(commonDtype: DType, opName: OpName): DType {
    let policy: TypePromotionKind;

    const op = getOpDef(opName);
    if (!op) {
        policy = 'match_input';
    } else {
        policy = deriveTypePromotionKindFromRule(op.dtype);
    }

    return resolveResultTypeWithPolicy(commonDtype, policy, opName);
}