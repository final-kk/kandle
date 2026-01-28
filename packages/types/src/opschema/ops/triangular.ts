/**
 * NN-Kit Operator Schema v7 - Triangular Matrix Operations
 *
 * 三角矩阵操作
 * - Kernel: triu, tril
 * - View: diagonal
 * - Composite: diag, trace
 *
 * @module v7/ops/triangular
 */

import { SchemaT, SchemaShape, SchemaDtype } from '../helpers';
import type { OpEntry } from '../types';

// ============================================================================
// triu - 上三角矩阵
// ============================================================================

export const triu: OpEntry = {
    name: 'triu',
    mechanism: 'Triangular',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量 (..., M, N)' },
            { name: 'diagonal', type: SchemaT.Scalar('int'), default: 0, doc: '对角线偏移' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'triu',
    doc: '上三角矩阵: 保留 row <= col + diagonal 的元素，其余置零',
    codegen: { tensorMethod: 'triu' },
};

// ============================================================================
// tril - 下三角矩阵
// ============================================================================

export const tril: OpEntry = {
    name: 'tril',
    mechanism: 'Triangular',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量 (..., M, N)' },
            { name: 'diagonal', type: SchemaT.Scalar('int'), default: 0, doc: '对角线偏移' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'tril',
    doc: '下三角矩阵: 保留 row >= col + diagonal 的元素，其余置零',
    codegen: { tensorMethod: 'tril' },
};

// ============================================================================
// diagonal - 对角线视图
// ============================================================================

export const diagonal: OpEntry = {
    name: 'diagonal',
    mechanism: 'View',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'offset', type: SchemaT.Scalar('int'), default: 0, doc: '对角线偏移' },
            { name: 'dim1', type: SchemaT.Axis(), default: 0, doc: '第一个维度' },
            { name: 'dim2', type: SchemaT.Axis(), default: 1, doc: '第二个维度' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('computed_view'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'diagonal',
    doc: '获取对角线视图 (Partial View)',
    codegen: { tensorMethod: 'diagonal' },
};

// ============================================================================
// diag - 对角线构造/提取
// ============================================================================

export const diag: OpEntry = {
    name: 'diag',
    mechanism: 'Composite',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量 (1D 或 2D)' },
            { name: 'diagonal', type: SchemaT.Scalar('int'), default: 0, doc: '对角线偏移' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('computed_diag'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'diag',
    doc: '对角线构造 (1D->2D) 或提取 (2D->1D)',
    codegen: { tensorMethod: 'diag' },
};

// ============================================================================
// trace - 矩阵迹
// ============================================================================

export const trace: OpEntry = {
    name: 'trace',
    mechanism: 'Composite',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量 (2D)' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('[]'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'trace',
    doc: '矩阵迹: 对角线元素之和',
    codegen: { tensorMethod: 'trace' },
};
