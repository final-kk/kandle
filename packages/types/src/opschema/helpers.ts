/**
 * NN-Kit Operator Schema v5 - Helper Functions
 *
 * 提供简洁的构建器 API: SchemaT, SchemaShape, SchemaDtype
 *
 * @module v5/helpers
 */

import type {
    TensorType,
    TensorListType,
    ScalarType,
    ScalarListType,
    ShapeType,
    AxisType,
    AxesType,
    DTypeValueType,
    DeviceType,
    MemoryFormatType,
    BoolType,
    StringType,
    OptionalType,
    UnionType,
    ValueType,
    BroadcastShapeRule,
    SameAsShapeRule,
    ReductionShapeRule,
    MatmulShapeRule,
    PermuteShapeRule,
    TransposeShapeRule,
    ExplicitShapeRule,
    PromoteDTypeRule,
    SameAsDTypeRule,
    FixedDTypeRule,
    ExplicitDTypeRule,
    SchemaDTypeCategory,
} from './types';

import type { DType } from '../base';

// ============================================================================
// Value Type Builder (T)
// ============================================================================

/**
 * Type Builder - 创建 ValueType 的工厂函数
 *
 * @example
 * ```ts
 * SchemaT.Tensor()                        // 任意 Tensor
 * SchemaT.Tensor({ dtype: 'float32' })    // float32 Tensor
 * SchemaT.Tensor({ ndim: 2 })             // 2D Tensor
 * SchemaT.Tensor({ ndim: { min: 1 } })    // 至少 1D Tensor
 * SchemaT.Scalar()                        // 任意标量
 * SchemaT.Scalar('int')                   // 整数标量
 * SchemaT.Optional(SchemaT.Tensor())      // 可选 Tensor
 * SchemaT.Union(SchemaT.Tensor(), SchemaT.Scalar())  // Tensor 或 Scalar
 * ```
 */
export const SchemaT = {
    Tensor: (opts?: Omit<TensorType, 'kind'>): TensorType => ({
        kind: 'Tensor',
        ...opts,
    }),

    TensorList: (): TensorListType => ({ kind: 'TensorList' }),

    Scalar: (numericKind?: ScalarType['numericKind']): ScalarType => ({
        kind: 'Scalar',
        numericKind,
    }),

    ScalarList: (numericKind?: ScalarListType['numericKind']): ScalarListType => ({
        kind: 'ScalarList',
        numericKind,
    }),

    Shape: (): ShapeType => ({ kind: 'Shape' }),

    Axis: (): AxisType => ({ kind: 'Axis' }),

    Axes: (): AxesType => ({ kind: 'Axes' }),

    DType: (): DTypeValueType => ({ kind: 'DType' }),

    Device: (): DeviceType => ({ kind: 'Device' }),

    MemoryFormat: (): MemoryFormatType => ({ kind: 'MemoryFormat' }),

    Bool: (): BoolType => ({ kind: 'Bool' }),

    String: (oneOf?: readonly string[]): StringType => ({
        kind: 'String',
        oneOf,
    }),

    Optional: (inner: ValueType): OptionalType => ({
        kind: 'Optional',
        inner,
    }),

    Union: (...types: ValueType[]): UnionType => ({
        kind: 'Union',
        types,
    }),
} as const;

// ============================================================================
// Shape Rule Builder
// ============================================================================

/**
 * Shape Rule Builder - 创建形状推导规则
 *
 * @example
 * ```ts
 * SchemaShape.broadcast('self', 'other')       // 广播
 * SchemaShape.same('self')                     // 相同形状
 * SchemaShape.reduction('self', 'dim', 'keepdim') // 归约
 * SchemaShape.matmul('self', 'other')          // 矩阵乘法
 * SchemaShape.permute('self', 'dims')          // 排列
 * SchemaShape.transpose('self', 'dim0', 'dim1') // 转置
 * SchemaShape.explicit('shape')                // 显式形状参数
 * SchemaShape.explicit('[]')                   // 标量输出
 * ```
 */
export const SchemaShape = {
    broadcast: (...inputs: string[]): BroadcastShapeRule => ({
        rule: 'broadcast',
        inputs,
    }),

    same: (as: string): SameAsShapeRule => ({
        rule: 'same',
        as,
    }),

    reduction: (input: string, axis: string, keepdims: string): ReductionShapeRule => ({
        rule: 'reduction',
        input,
        axis,
        keepdims,
    }),

    matmul: (left: string, right: string): MatmulShapeRule => ({
        rule: 'matmul',
        left,
        right,
    }),

    permute: (input: string, dims: string): PermuteShapeRule => ({
        rule: 'permute',
        input,
        dims,
    }),

    transpose: (input: string, dim0: string, dim1: string): TransposeShapeRule => ({
        rule: 'transpose',
        input,
        dim0,
        dim1,
    }),

    explicit: (expr: string): ExplicitShapeRule => ({
        rule: 'explicit',
        expr,
    }),
} as const;

// ============================================================================
// DType Rule Builder
// ============================================================================

/**
 * DType Rule Builder - 创建类型推导规则
 *
 * @example
 * ```ts
 * SchemaDtype.promote(['self', 'other'])               // 类型提升
 * SchemaDtype.promote(['self'], 'Floating')            // 提升到浮点
 * SchemaDtype.promote(['self'], 'SignedIntegral')      // 提升到有符号整数
 * SchemaDtype.same('self')                             // 相同类型
 * SchemaDtype.fixed('bool')                            // 固定 bool
 * SchemaDtype.fixed('int64')                           // 固定 int64
 * SchemaDtype.explicit('dtype')                        // 显式类型参数
 * SchemaDtype.explicit('dtype', 'self')                // 显式或回退到 self
 * ```
 */
export const SchemaDtype = {
    promote: (inputs: readonly string[], toCategory?: SchemaDTypeCategory): PromoteDTypeRule => ({
        rule: 'promote',
        inputs,
        toCategory,
    }),

    same: (as: string): SameAsDTypeRule => ({
        rule: 'same',
        as,
    }),

    fixed: (dtype: FixedDTypeRule['dtype']): FixedDTypeRule => ({
        rule: 'fixed',
        dtype,
    }),

    explicit: (param: string, fallback?: string): ExplicitDTypeRule => ({
        rule: 'explicit',
        param,
        fallback,
    }),
} as const;
