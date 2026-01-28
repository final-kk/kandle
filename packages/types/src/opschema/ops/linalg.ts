/**
 * NN-Kit Operator Schema v7 - Linear Algebra Operations
 *
 * 线性代数运算
 * Mechanism: Kernel / Composite
 *
 * @module v7/ops/linalg
 */

import { SchemaT, SchemaShape, SchemaDtype } from '../helpers';
import type { OpEntry } from '../types';

// ============================================================================
// matmul (通用)
// ============================================================================

export const matmul: OpEntry = {
    name: 'matmul',
    mechanism: 'Matrix',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor({ ndim: { min: 1 } }), doc: '第一个输入张量' },
            { name: 'other', type: SchemaT.Tensor({ ndim: { min: 1 } }), doc: '第二个输入张量' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.matmul('self', 'other'),
    dtype: SchemaDtype.promote(['self', 'other']),
    dispatchKey: 'matmul',
    doc: '矩阵乘法 (支持批量)',
    codegen: { tensorMethod: 'matmul' },
};

// ============================================================================
// mm (严格 2D)
// ============================================================================

export const mm: OpEntry = {
    name: 'mm',
    mechanism: 'Matrix',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor({ ndim: 2 }), doc: '第一个 2D 矩阵' },
            { name: 'mat2', type: SchemaT.Tensor({ ndim: 2 }), doc: '第二个 2D 矩阵' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.matmul('self', 'mat2'),
    dtype: SchemaDtype.promote(['self', 'mat2']),
    dispatchKey: 'mm',
    doc: '2D 矩阵乘法',
    codegen: { tensorMethod: 'mm' },
    kernelConfig: {
        tensorMap: { mat2: 'other' }
    }
};

// ============================================================================
// bmm (批量)
// ============================================================================

export const bmm: OpEntry = {
    name: 'bmm',
    mechanism: 'Matrix',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor({ ndim: 3 }), doc: '第一个 3D 批量矩阵' },
            { name: 'mat2', type: SchemaT.Tensor({ ndim: 3 }), doc: '第二个 3D 批量矩阵' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.matmul('self', 'mat2'),
    dtype: SchemaDtype.promote(['self', 'mat2']),
    dispatchKey: 'bmm',
    doc: '批量矩阵乘法',
    codegen: { tensorMethod: 'bmm' },
};

// ============================================================================
// mv (矩阵-向量)
// ============================================================================

export const mv: OpEntry = {
    name: 'mv',
    mechanism: 'Matrix',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor({ ndim: 2 }), doc: '2D 矩阵' },
            { name: 'vec', type: SchemaT.Tensor({ ndim: 1 }), doc: '1D 向量' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.matmul('self', 'vec'),
    dtype: SchemaDtype.promote(['self', 'vec']),
    dispatchKey: 'mv',
    doc: '矩阵-向量乘法',
    codegen: { tensorMethod: 'mv' },
};

// ============================================================================
// dot (向量点积)
// ============================================================================

export const dot: OpEntry = {
    name: 'dot',
    mechanism: 'Matrix',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor({ ndim: 1 }), doc: '第一个 1D 向量' },
            { name: 'other', type: SchemaT.Tensor({ ndim: 1 }), doc: '第二个 1D 向量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('[]'),
    dtype: SchemaDtype.promote(['self', 'other']),
    dispatchKey: 'dot',
    doc: '向量点积',
    codegen: { tensorMethod: 'dot' },
};

// ============================================================================
// addmm (GEMM)
// ============================================================================

export const addmm: OpEntry = {
    name: 'addmm',
    mechanism: 'Matrix',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor({ ndim: 2 }), doc: '偏置矩阵' },
            { name: 'mat1', type: SchemaT.Tensor({ ndim: 2 }), doc: '第一个矩阵' },
            { name: 'mat2', type: SchemaT.Tensor({ ndim: 2 }), doc: '第二个矩阵' },
            { name: 'beta', type: SchemaT.Scalar(), default: 1, doc: 'self 的系数' },
            { name: 'alpha', type: SchemaT.Scalar(), default: 1, doc: 'mat1 @ mat2 的系数' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.matmul('mat1', 'mat2'),
    dtype: SchemaDtype.promote(['self', 'mat1', 'mat2']),
    dispatchKey: 'addmm',
    doc: 'out = beta * self + alpha * (mat1 @ mat2)',
    codegen: { tensorMethod: 'addmm' },
};

// ============================================================================
// addmv (GEMV)
// ============================================================================

export const addmv: OpEntry = {
    name: 'addmv',
    mechanism: 'Matrix',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor({ ndim: 1 }), doc: '偏置向量' },
            { name: 'mat', type: SchemaT.Tensor({ ndim: 2 }), doc: '矩阵' },
            { name: 'vec', type: SchemaT.Tensor({ ndim: 1 }), doc: '向量' },
            { name: 'beta', type: SchemaT.Scalar(), default: 1, doc: 'self 的系数' },
            { name: 'alpha', type: SchemaT.Scalar(), default: 1, doc: 'mat @ vec 的系数' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.matmul('mat', 'vec'),
    dtype: SchemaDtype.promote(['self', 'mat', 'vec']),
    dispatchKey: 'addmv',
    doc: 'out = beta * self + alpha * (mat @ vec)',
    codegen: { tensorMethod: 'addmv' },
};

// ============================================================================
// outer (外积)
// ============================================================================

export const outer: OpEntry = {
    name: 'outer',
    mechanism: 'Matrix',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor({ ndim: 1 }), doc: '第一个向量' },
            { name: 'vec2', type: SchemaT.Tensor({ ndim: 1 }), doc: '第二个向量' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('[self.shape[0], vec2.shape[0]]'),
    dtype: SchemaDtype.promote(['self', 'vec2']),
    dispatchKey: 'outer',
    doc: '向量外积',
    codegen: { tensorMethod: 'outer' },
};

// ============================================================================
// baddbmm (批量 GEMM)
// ============================================================================

export const baddbmm: OpEntry = {
    name: 'baddbmm',
    mechanism: 'Matrix',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor({ ndim: 3 }), doc: '偏置批量矩阵' },
            { name: 'batch1', type: SchemaT.Tensor({ ndim: 3 }), doc: '第一个批量矩阵' },
            { name: 'batch2', type: SchemaT.Tensor({ ndim: 3 }), doc: '第二个批量矩阵' },
            { name: 'beta', type: SchemaT.Scalar(), default: 1, doc: 'self 的系数' },
            { name: 'alpha', type: SchemaT.Scalar(), default: 1, doc: 'batch1 @ batch2 的系数' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.matmul('batch1', 'batch2'),
    dtype: SchemaDtype.promote(['self', 'batch1', 'batch2']),
    dispatchKey: 'baddbmm',
    doc: 'out = beta * self + alpha * (batch1 @ batch2)',
    codegen: { tensorMethod: 'baddbmm' },
    kernelConfig: { argMap: { beta: 'beta', alpha: 'alpha' } }
};

// ============================================================================
// linear (F.linear)
// ============================================================================

export const linear: OpEntry = {
    name: 'linear',
    mechanism: 'Composite',
    signature: {
        params: [
            { name: 'input', type: SchemaT.Tensor({ ndim: { min: 1 } }), doc: '输入张量 (..., in_features)' },
            { name: 'weight', type: SchemaT.Tensor({ ndim: 2 }), doc: '权重矩阵 (out_features, in_features)' },
            { name: 'bias', type: SchemaT.Optional(SchemaT.Tensor({ ndim: 1 })), doc: '偏置向量 (out_features)' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('linear(input.shape, weight.shape)'),
    dtype: SchemaDtype.promote(['input', 'weight']),
    dispatchKey: 'linear',
    doc: '线性变换: y = input @ weight.T + bias',
    codegen: {
        tensorMethod: false,
        namespace: 'nn.functional'
    },
};
