/**
 * NN-Kit Operator Schema v6 - Core Type Definitions
 *
 * v6 架构核心变化:
 * - 引入 DispatchMechanism (Iterator, Kernel, Composite) 替代 ComputePattern
 * - 引入 KernelConfig 和 IteratorType
 * - 明确分离执行机制与语义分类
 *
 * @module v6/types
 */

import type { DType } from '../base';

// ============================================================================
// Dispatch Mechanism (v6)
// ============================================================================

/**
 * Dispatch Mechanism - 决定操作符的执行方式
 * 
 * - Iterator: 使用 TensorIterator 进行广播和迭代 (Pointwise, Reduction, Scan)
 * - Kernel: 直接调用后端 Kernel (Matrix, Conv, Norm, Sort, etc.)
 * - Composite: 组合操作，由 JS 实现调用其他算子
 * - View: 纯元数据操作
 * - Copy: 数据复制/转换
 * - Factory: 创建新 Tensor
 */
export type OpMechanism =
    | 'Iterator'    // TensorIterator-based: Pointwise/Reduction/Scan
    | 'Composite'   // JS 组合实现 (如 linear, trace, embedding)
    | 'View'        // 纯元数据操作 (reshape, permute, etc.)
    | 'Copy'        // 数据复制/转换 (clone, contiguous, to)
    | 'Factory'     // 创建新 Tensor (zeros, ones, rand, etc.)
    // 专用 Kernel 机制 (替代原来的通用 'Kernel'):
    | 'Matrix'      // 矩阵操作: matmul, mm, bmm, addmm, etc.
    | 'Window'      // 窗口操作: conv, pool
    | 'WindowFunc'  // 窗函数生成: hann, kaiser, etc. (信号处理)
    | 'Normalize'   // 归一化: softmax, layer_norm, batch_norm, etc.
    | 'Sort'        // 排序: sort, argsort, topk
    | 'Gather'      // 索引读取: index_select
    | 'Scatter'     // 索引写入: scatter, scatter_add, scatter_reduce
    | 'Triangular'  // 三角操作: triu, tril
    | 'Shape'       // 形状操作(需拷贝): cat, stack
    | 'FFT';        // FFT 操作: fft, ifft, rfft, irfft

/**
 * Iterator Type - 细分 Iterator 的行为
 */
export type IteratorType =
    | 'Map'         // Pointwise-like (Unary, Binary, Ternary)
    | 'Reduce'      // Reduction
    | 'Scan';       // Scan (Prefix Sum)

// ============================================================================
// DType Category
// ============================================================================

/**
 * Schema DType Category - 用于 OpEntry 中的类型约束和类型提升
 * 与 base.ts 中的 DTypeCategory enum 不同，这是 opschema 专用的字符串联合类型
 */
export type SchemaDTypeCategory =
    | 'All'
    | 'Numeric'
    | 'Real'
    | 'Floating'
    | 'Integral'
    | 'SignedIntegral'
    | 'UnsignedIntegral'
    | 'Complex'
    | 'Bool';

// ============================================================================
// Value Types (Discriminated Union)
// ============================================================================

/**
 * Value Type - 参数可接受的值类型
 */
export type ValueType =
    // Tensor Types
    | TensorType
    | TensorListType
    // Scalar Types
    | ScalarType
    | ScalarListType
    // Metadata Types
    | ShapeType
    | AxisType
    | AxesType
    | DTypeValueType
    | DeviceType
    | MemoryFormatType
    | BoolType
    | StringType
    // Composite Types
    | OptionalType
    | UnionType;

export interface TensorType {
    readonly kind: 'Tensor';
    readonly dtype?: DType | SchemaDTypeCategory | readonly DType[];
    readonly ndim?: number | { readonly min?: number; readonly max?: number };
}

export interface TensorListType {
    readonly kind: 'TensorList';
    readonly numericKind?: 'int' | 'float' | 'bool' | 'complex' | 'number';
}

export interface ScalarType {
    readonly kind: 'Scalar';
    readonly numericKind?: 'int' | 'float' | 'bool' | 'complex' | 'number';
}

export interface ScalarListType {
    readonly kind: 'ScalarList';
    readonly numericKind?: 'int' | 'float';
}

export interface ShapeType {
    readonly kind: 'Shape';
}

export interface AxisType {
    readonly kind: 'Axis';
}

export interface AxesType {
    readonly kind: 'Axes';
}

export interface DTypeValueType {
    readonly kind: 'DType';
}

export interface DeviceType {
    readonly kind: 'Device';
}

export interface MemoryFormatType {
    readonly kind: 'MemoryFormat';
}

export interface BoolType {
    readonly kind: 'Bool';
}

export interface StringType {
    readonly kind: 'String';
    readonly oneOf?: readonly string[];
}

export interface OptionalType {
    readonly kind: 'Optional';
    readonly inner: ValueType;
}

export interface UnionType {
    readonly kind: 'Union';
    readonly types: readonly ValueType[];
}

// ============================================================================
// Shape Inference Rules
// ============================================================================

export type ShapeRule =
    | BroadcastShapeRule
    | SameAsShapeRule
    | ReductionShapeRule
    | MatmulShapeRule
    | PermuteShapeRule
    | TransposeShapeRule
    | ExplicitShapeRule;

export interface BroadcastShapeRule {
    readonly rule: 'broadcast';
    readonly inputs: readonly string[];
}

export interface SameAsShapeRule {
    readonly rule: 'same';
    readonly as: string;
}

export interface ReductionShapeRule {
    readonly rule: 'reduction';
    readonly input: string;
    readonly axis: string;
    readonly keepdims: string;
}

export interface MatmulShapeRule {
    readonly rule: 'matmul';
    readonly left: string;
    readonly right: string;
}

export interface PermuteShapeRule {
    readonly rule: 'permute';
    readonly input: string;
    readonly dims: string;
}

export interface TransposeShapeRule {
    readonly rule: 'transpose';
    readonly input: string;
    readonly dim0: string;
    readonly dim1: string;
}

export interface ExplicitShapeRule {
    readonly rule: 'explicit';
    readonly expr: string;
}

// ============================================================================
// DType Inference Rules
// ============================================================================

export type DTypeRule =
    | PromoteDTypeRule
    | SameAsDTypeRule
    | FixedDTypeRule
    | ExplicitDTypeRule;

export interface PromoteDTypeRule {
    readonly rule: 'promote';
    readonly inputs: readonly string[];
    readonly toCategory?: SchemaDTypeCategory;
}

export interface SameAsDTypeRule {
    readonly rule: 'same';
    readonly as: string;
}

export interface FixedDTypeRule {
    readonly rule: 'fixed';
    readonly dtype: DType | 'bool' | 'int64' | 'float32' | 'float64';
}

export interface ExplicitDTypeRule {
    readonly rule: 'explicit';
    readonly param: string;
    readonly fallback?: string;
}

// ============================================================================
// Parameter Definition
// ============================================================================

export interface ParamDef {
    readonly name: string;
    readonly type: ValueType;
    readonly default?: string | number | boolean;
    readonly doc?: string;
}

// ============================================================================
// Return Definition
// ============================================================================

export type ReturnDef =
    | { readonly single: ValueType }
    | {
        readonly tuple: readonly {
            readonly name?: string;
            readonly type: ValueType;
        }[];
    };

// ============================================================================
// Operator Signature
// ============================================================================

export interface OpSignature {
    readonly params: readonly ParamDef[];
    readonly returns: ReturnDef;
}

// ============================================================================
// Configurations
// ============================================================================

/**
 * Iterator Configuration - 用于 mechanism = 'Iterator'
 */
export interface IteratorConfig {
    /**
     * 工厂方法类型 (保留 v5 字段以兼容，但 v6 主要看 iteratorType)
     * - unary: 单输入
     * - binary: 双输入
     * - ternary: 三输入
     * - reduction: 归约
     * - scan: 扫描
     */
    readonly factory: 'unary' | 'binary' | 'ternary' | 'reduction' | 'scan';

    /**
     * 参与 Iterator 迭代的 Tensor 参数名
     */
    readonly tensorInputs: readonly string[];

    /**
     * 传递给 kernel 的 Scalar 参数名
     */
    readonly scalarArgs: readonly string[];

    /**
     * 输出参数名
     */
    readonly outputs?: readonly string[];
}

/**
 * Kernel Configuration - 用于 mechanism = 'Kernel'
 * 允许重命名参数以匹配后端 Kernel 的要求
 */
export interface KernelConfig {
    /**
     * Tensor 参数映射: { schemaParamName: backendArgName }
     * 如果未指定，默认使用参数名
     */
    readonly tensorMap?: Record<string, string>;

    /**
     * Scalar/Metadata 参数映射: { schemaParamName: backendArgName }
     * 如果未指定，默认使用参数名
     */
    readonly argMap?: Record<string, string>;
}

/**
 * Codegen Configuration
 */
export interface CodegenConfig {
    /**
     * Tensor 方法名
     * - string: 方法名
     * - false: 不生成 Tensor 方法
     * - undefined: 使用操作符名
     */
    readonly tensorMethod?: string | false;

    /**
     * 作为 this 的参数名 (默认第一个 Tensor 参数)
     */
    readonly thisArg?: string;

    /**
     * 是否是静态方法 (如 Tensor.zeros)
     */
    readonly staticMethod?: boolean;

    /**
     * 命名空间 (如 'nn.functional', 'fft')
     */
    readonly namespace?: string;

    /**
     * 命名空间键别名
     * 
     * 当函数名与 namespace 名冲突时使用。
     * 设置后，函数将生成为 `{name}Impl`，在 namespace 对象中使用此别名作为 key。
     * 
     * 例如: fft 函数在 fft namespace 中
     * - namespaceKeyAlias: 'fft'
     * - 生成函数: fftImpl
     * - namespace 对象: { fft: fftImpl, ... }
     */
    readonly namespaceKeyAlias?: string;

    /**
     * 条件返回类型配置
     */
    readonly conditionalReturn?: {
        readonly param: string;
        readonly tupleSize: number;
    };
}

// ============================================================================
// OpEntry - v6 核心类型
// ============================================================================

/**
 * OpEntry - 单个操作符变体的完整定义
 */
export interface OpEntry {
    /**
     * 操作符名称
     */
    readonly name: string;

    /**
     * 变体名称
     */
    readonly variant?: string;

    /**
     * 分发机制 (v6 核心变更)
     */
    readonly mechanism: OpMechanism;

    /**
     * Iterator 类型 (仅当 mechanism === 'Iterator' 时需要)
     */
    readonly iteratorType?: IteratorType;

    /**
     * 函数签名
     */
    readonly signature: OpSignature;

    /**
     * 迭代器配置 (仅 mechanism === 'Iterator' 需要)
     */
    readonly iteratorConfig?: IteratorConfig;

    /**
     * Kernel 配置 (仅 mechanism === 'Kernel' 可选)
     */
    readonly kernelConfig?: KernelConfig;

    /**
     * 形状推导规则
     */
    readonly shape: ShapeRule;

    /**
     * 类型推导规则
     */
    readonly dtype: DTypeRule;

    /**
     * Dispatch Key
     * - 对于 Kernel: 后端 Kernel 名称
     * - 对于 Composite: 组合函数注册名
     * - 对于 Iterator: 算子名称
     */
    readonly dispatchKey: string;

    /**
     * 文档说明
     */
    readonly doc?: string;

    /**
     * CodeGen 配置
     */
    readonly codegen?: CodegenConfig;
}
