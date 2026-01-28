/**
 * NN-Kit Operator Schema
 *
 * 核心架构设计:
 * - 每个签名变体是独立的 OpEntry (不再是 signatures 数组)
 * - IteratorConfig 明确声明 Tensor/Scalar 参数分离
 * - CodeGen 生成编译时确定的 typeof 分支
 * - Pattern Handler 按计算模式分发执行
 *
 * @example
 * ```ts
 * import { SchemaT, SchemaShape, SchemaDtype, type OpEntry } from '@kandle/types/opschema';
 *
 * export const add_Tensor: OpEntry = {
 *     name: 'add',
 *     variant: 'Tensor',
 *     pattern: 'Pointwise',
 *     signature: {
 *         params: [
 *             { name: 'self', type: SchemaT.Tensor() },
 *             { name: 'other', type: SchemaT.Tensor() },
 *             { name: 'alpha', type: SchemaT.Scalar(), default: 1 },
 *         ],
 *         returns: { single: SchemaT.Tensor() },
 *     },
 *     iteratorConfig: {
 *         factory: 'binary',
 *         tensorInputs: ['self', 'other'],
 *         scalarArgs: ['alpha'],
 *     },
 *     shape: SchemaShape.broadcast('self', 'other'),
 *     dtype: SchemaDtype.promote(['self', 'other']),
 *     dispatchKey: 'add',
 * };
 * ```
 *
 * @module opschema
 */

// ============================================================================
// Type Exports
// ============================================================================

// ============================================================================
// Type Exports
// ============================================================================

export type {
    // Mechanisms
    OpMechanism,
    SchemaDTypeCategory,

    // Value Types
    ValueType,
    TensorType,
    TensorListType,
    ScalarType,
    ScalarListType,
    ShapeType,
    AxisType,
    AxesType,
    DTypeValueType,
    DeviceType,
    BoolType,
    StringType,
    OptionalType,
    UnionType,

    // Shape Rules
    ShapeRule,
    BroadcastShapeRule,
    SameAsShapeRule,
    ReductionShapeRule,
    MatmulShapeRule,
    PermuteShapeRule,
    TransposeShapeRule,
    ExplicitShapeRule,

    // DType Rules
    DTypeRule,
    PromoteDTypeRule,
    SameAsDTypeRule,
    FixedDTypeRule,
    ExplicitDTypeRule,

    // Parameter & Return
    ParamDef,
    ReturnDef,
    OpSignature,

    // Core Types
    IteratorConfig,
    KernelConfig,
    CodegenConfig,
    OpEntry,
} from './types';

// ============================================================================
// Helper Exports
// ============================================================================

export { SchemaT, SchemaShape, SchemaDtype } from './helpers';

// ============================================================================
// Registry Exports
// ============================================================================

export { OpRegistry, getOpEntry, getOpVariants, getOpsByMechanism, getAllOpNames, type OpName, MechanismGroups } from './registry';

// ============================================================================
// Ops Exports (for direct access to op definitions)
// ============================================================================

export * as ops from './ops';
