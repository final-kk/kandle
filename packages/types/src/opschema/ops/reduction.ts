/**
 * NN-Kit Operator Schema v6 - Reduction Operations
 *
 * 归约运算符
 * Mechanism: Iterator (Reduce)
 *
 * @module v6/ops/reduction
 */

import { SchemaT, SchemaShape, SchemaDtype } from '../helpers';
import type { OpEntry } from '../types';

// ============================================================================
// sum
// ============================================================================

export const sum: OpEntry = {
    name: 'sum',
    mechanism: 'Iterator',
    iteratorType: 'Reduce',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'dim', type: SchemaT.Optional(SchemaT.Axes()), doc: '归约维度' },
            { name: 'keepdim', type: SchemaT.Bool(), default: false, doc: '是否保持维度' },
            { name: 'dtype', type: SchemaT.Optional(SchemaT.DType()), doc: '输出类型' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'reduction',
        tensorInputs: ['self'],
        scalarArgs: ['dim', 'keepdim', 'dtype'],
    },
    shape: SchemaShape.reduction('self', 'dim', 'keepdim'),
    dtype: SchemaDtype.promote(['self'], 'SignedIntegral'),
    dispatchKey: 'sum',
    doc: '沿维度求和',
    codegen: { tensorMethod: 'sum' },
};

// ============================================================================
// nansum
// ============================================================================

export const nansum: OpEntry = {
    name: 'nansum',
    mechanism: 'Iterator',
    iteratorType: 'Reduce',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'dim', type: SchemaT.Optional(SchemaT.Axes()), doc: '归约维度' },
            { name: 'keepdim', type: SchemaT.Bool(), default: false, doc: '是否保持维度' },
            { name: 'dtype', type: SchemaT.Optional(SchemaT.DType()), doc: '输出类型' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'reduction',
        tensorInputs: ['self'],
        scalarArgs: ['dim', 'keepdim', 'dtype'],
    },
    shape: SchemaShape.reduction('self', 'dim', 'keepdim'),
    dtype: SchemaDtype.promote(['self'], 'SignedIntegral'),
    dispatchKey: 'nansum',
    doc: '沿维度求和(忽略NaN)',
    codegen: { tensorMethod: 'nansum' },
};

// ============================================================================
// mean
// ============================================================================

export const mean: OpEntry = {
    name: 'mean',
    mechanism: 'Iterator',
    iteratorType: 'Reduce',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor({ dtype: 'Floating' }), doc: '输入张量' },
            { name: 'dim', type: SchemaT.Optional(SchemaT.Axes()), doc: '归约维度' },
            { name: 'keepdim', type: SchemaT.Bool(), default: false, doc: '是否保持维度' },
            { name: 'dtype', type: SchemaT.Optional(SchemaT.DType()), doc: '输出类型' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'reduction',
        tensorInputs: ['self'],
        scalarArgs: ['dim', 'keepdim', 'dtype'],
    },
    shape: SchemaShape.reduction('self', 'dim', 'keepdim'),
    dtype: SchemaDtype.promote(['self'], 'Floating'),
    dispatchKey: 'mean',
    doc: '沿维度求均值',
    codegen: { tensorMethod: 'mean' },
};

// ============================================================================
// nanmean
// ============================================================================

export const nanmean: OpEntry = {
    name: 'nanmean',
    mechanism: 'Iterator',
    iteratorType: 'Reduce',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor({ dtype: 'Floating' }), doc: '输入张量' },
            { name: 'dim', type: SchemaT.Optional(SchemaT.Axes()), doc: '归约维度' },
            { name: 'keepdim', type: SchemaT.Bool(), default: false, doc: '是否保持维度' },
            { name: 'dtype', type: SchemaT.Optional(SchemaT.DType()), doc: '输出类型' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'reduction',
        tensorInputs: ['self'],
        scalarArgs: ['dim', 'keepdim', 'dtype'],
    },
    shape: SchemaShape.reduction('self', 'dim', 'keepdim'),
    dtype: SchemaDtype.promote(['self'], 'Floating'),
    dispatchKey: 'nanmean',
    doc: '沿维度求均值(忽略NaN)',
    codegen: { tensorMethod: 'nanmean' },
};

// ============================================================================
// prod
// ============================================================================

export const prod: OpEntry = {
    name: 'prod',
    mechanism: 'Iterator',
    iteratorType: 'Reduce',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'dim', type: SchemaT.Optional(SchemaT.Axis()), doc: '归约维度' },
            { name: 'keepdim', type: SchemaT.Bool(), default: false, doc: '是否保持维度' },
            { name: 'dtype', type: SchemaT.Optional(SchemaT.DType()), doc: '输出类型' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'reduction',
        tensorInputs: ['self'],
        scalarArgs: ['dim', 'keepdim', 'dtype'],
    },
    shape: SchemaShape.reduction('self', 'dim', 'keepdim'),
    dtype: SchemaDtype.promote(['self'], 'SignedIntegral'),
    dispatchKey: 'prod',
    doc: '沿维度求积',
    codegen: { tensorMethod: 'prod' },
};

// ============================================================================
// max (global)
// ============================================================================

export const max_global: OpEntry = {
    name: 'max',
    variant: 'global',
    mechanism: 'Iterator',
    iteratorType: 'Reduce',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'reduction',
        tensorInputs: ['self'],
        scalarArgs: [],
    },
    shape: SchemaShape.explicit('[]'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'max',
    doc: '全局最大值',
    codegen: { tensorMethod: 'max' },
};

// ============================================================================
// max (dim)
// ============================================================================

export const max_dim: OpEntry = {
    name: 'max',
    variant: 'dim',
    mechanism: 'Iterator',
    iteratorType: 'Reduce',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'dim', type: SchemaT.Axis(), doc: '归约维度' },
            { name: 'keepdim', type: SchemaT.Bool(), default: false, doc: '是否保持维度' },
        ],
        returns: {
            tuple: [
                { name: 'values', type: SchemaT.Tensor() },
                { name: 'indices', type: SchemaT.Tensor({ dtype: 'int64' }) },
            ],
        },
    },
    iteratorConfig: {
        factory: 'reduction',
        tensorInputs: ['self'],
        scalarArgs: ['dim', 'keepdim'],
    },
    shape: SchemaShape.reduction('self', 'dim', 'keepdim'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'max_dim',
    doc: '沿维度最大值及索引',
    codegen: { tensorMethod: 'max' },
};

// ============================================================================
// min (global)
// ============================================================================

export const min_global: OpEntry = {
    name: 'min',
    variant: 'global',
    mechanism: 'Iterator',
    iteratorType: 'Reduce',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'reduction',
        tensorInputs: ['self'],
        scalarArgs: [],
    },
    shape: SchemaShape.explicit('[]'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'min',
    doc: '全局最小值',
    codegen: { tensorMethod: 'min' },
};

// ============================================================================
// min (dim)
// ============================================================================

export const min_dim: OpEntry = {
    name: 'min',
    variant: 'dim',
    mechanism: 'Iterator',
    iteratorType: 'Reduce',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'dim', type: SchemaT.Axis(), doc: '归约维度' },
            { name: 'keepdim', type: SchemaT.Bool(), default: false, doc: '是否保持维度' },
        ],
        returns: {
            tuple: [
                { name: 'values', type: SchemaT.Tensor() },
                { name: 'indices', type: SchemaT.Tensor({ dtype: 'int64' }) },
            ],
        },
    },
    iteratorConfig: {
        factory: 'reduction',
        tensorInputs: ['self'],
        scalarArgs: ['dim', 'keepdim'],
    },
    shape: SchemaShape.reduction('self', 'dim', 'keepdim'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'min_dim',
    doc: '沿维度最小值及索引',
    codegen: { tensorMethod: 'min' },
};

// ============================================================================
// argmax / argmin
// ============================================================================

export const argmax: OpEntry = {
    name: 'argmax',
    mechanism: 'Iterator',
    iteratorType: 'Reduce',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'dim', type: SchemaT.Optional(SchemaT.Axis()), doc: '归约维度' },
            { name: 'keepdim', type: SchemaT.Bool(), default: false, doc: '是否保持维度' },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'int64' }) },
    },
    iteratorConfig: {
        factory: 'reduction',
        tensorInputs: ['self'],
        scalarArgs: ['dim', 'keepdim'],
    },
    shape: SchemaShape.reduction('self', 'dim', 'keepdim'),
    dtype: SchemaDtype.fixed('int64'),
    dispatchKey: 'argmax',
    doc: '沿维度最大值索引',
    codegen: { tensorMethod: 'argmax' },
};

export const argmin: OpEntry = {
    name: 'argmin',
    mechanism: 'Iterator',
    iteratorType: 'Reduce',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'dim', type: SchemaT.Optional(SchemaT.Axis()), doc: '归约维度' },
            { name: 'keepdim', type: SchemaT.Bool(), default: false, doc: '是否保持维度' },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'int64' }) },
    },
    iteratorConfig: {
        factory: 'reduction',
        tensorInputs: ['self'],
        scalarArgs: ['dim', 'keepdim'],
    },
    shape: SchemaShape.reduction('self', 'dim', 'keepdim'),
    dtype: SchemaDtype.fixed('int64'),
    dispatchKey: 'argmin',
    doc: '沿维度最小值索引',
    codegen: { tensorMethod: 'argmin' },
};

// ============================================================================
// all / any
// ============================================================================

export const all: OpEntry = {
    name: 'all',
    mechanism: 'Iterator',
    iteratorType: 'Reduce',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'dim', type: SchemaT.Optional(SchemaT.Axis()), doc: '归约维度' },
            { name: 'keepdim', type: SchemaT.Bool(), default: false, doc: '是否保持维度' },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'bool' }) },
    },
    iteratorConfig: {
        factory: 'reduction',
        tensorInputs: ['self'],
        scalarArgs: ['dim', 'keepdim'],
    },
    shape: SchemaShape.reduction('self', 'dim', 'keepdim'),
    dtype: SchemaDtype.fixed('bool'),
    dispatchKey: 'all',
    doc: '沿维度逻辑与',
    codegen: { tensorMethod: 'all' },
};

export const any: OpEntry = {
    name: 'any',
    mechanism: 'Iterator',
    iteratorType: 'Reduce',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'dim', type: SchemaT.Optional(SchemaT.Axis()), doc: '归约维度' },
            { name: 'keepdim', type: SchemaT.Bool(), default: false, doc: '是否保持维度' },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'bool' }) },
    },
    iteratorConfig: {
        factory: 'reduction',
        tensorInputs: ['self'],
        scalarArgs: ['dim', 'keepdim'],
    },
    shape: SchemaShape.reduction('self', 'dim', 'keepdim'),
    dtype: SchemaDtype.fixed('bool'),
    dispatchKey: 'any',
    doc: '沿维度逻辑或',
    codegen: { tensorMethod: 'any' },
};

// ============================================================================
// variance / std
// ============================================================================

export const variance: OpEntry = {
    name: 'variance',
    mechanism: 'Iterator',
    iteratorType: 'Reduce',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'dim', type: SchemaT.Optional(SchemaT.Axes()), doc: '归约维度' },
            { name: 'correction', type: SchemaT.Scalar('int'), default: 1, doc: '贝塞尔校正' },
            { name: 'keepdim', type: SchemaT.Bool(), default: false, doc: '是否保持维度' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'reduction',
        tensorInputs: ['self'],
        scalarArgs: ['dim', 'correction', 'keepdim'],
    },
    shape: SchemaShape.reduction('self', 'dim', 'keepdim'),
    dtype: SchemaDtype.promote(['self'], 'Floating'),
    dispatchKey: 'variance',
    doc: '沿维度方差',
    codegen: { tensorMethod: 'variance' },
};

export const std: OpEntry = {
    name: 'std',
    mechanism: 'Iterator',
    iteratorType: 'Reduce',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'dim', type: SchemaT.Optional(SchemaT.Axes()), doc: '归约维度' },
            { name: 'correction', type: SchemaT.Scalar('int'), default: 1, doc: '贝塞尔校正' },
            { name: 'keepdim', type: SchemaT.Bool(), default: false, doc: '是否保持维度' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'reduction',
        tensorInputs: ['self'],
        scalarArgs: ['dim', 'correction', 'keepdim'],
    },
    shape: SchemaShape.reduction('self', 'dim', 'keepdim'),
    dtype: SchemaDtype.promote(['self'], 'Floating'),
    dispatchKey: 'std',
    doc: '沿维度标准差',
    codegen: { tensorMethod: 'std' },
};

// ============================================================================
// norm
// ============================================================================

export const norm: OpEntry = {
    name: 'norm',
    mechanism: 'Iterator',
    iteratorType: 'Reduce',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'p', type: SchemaT.Scalar(), default: 2, doc: '范数阶数' },
            { name: 'dim', type: SchemaT.Optional(SchemaT.Axes()), doc: '归约维度' },
            { name: 'keepdim', type: SchemaT.Bool(), default: false, doc: '是否保持维度' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'reduction',
        tensorInputs: ['self'],
        scalarArgs: ['p', 'dim', 'keepdim'],
    },
    shape: SchemaShape.reduction('self', 'dim', 'keepdim'),
    dtype: SchemaDtype.promote(['self'], 'Floating'),
    dispatchKey: 'norm',
    doc: '沿维度范数',
    codegen: { tensorMethod: 'norm' },
};

// ============================================================================
// logsumexp
// ============================================================================

export const logsumexp: OpEntry = {
    name: 'logsumexp',
    mechanism: 'Iterator',
    iteratorType: 'Reduce',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'dim', type: SchemaT.Axes(), doc: '归约维度' },
            { name: 'keepdim', type: SchemaT.Bool(), default: false, doc: '是否保持维度' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'reduction',
        tensorInputs: ['self'],
        scalarArgs: ['dim', 'keepdim'],
    },
    shape: SchemaShape.reduction('self', 'dim', 'keepdim'),
    dtype: SchemaDtype.promote(['self'], 'Floating'),
    dispatchKey: 'logsumexp',
    doc: '数值稳定的 log(sum(exp(x)))，公式: max(x) + log(sum(exp(x - max(x))))',
    codegen: { tensorMethod: 'logsumexp' },
};
