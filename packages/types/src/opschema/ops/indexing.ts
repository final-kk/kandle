/**
 * NN-Kit Operator Schema v7 - Indexing Operations
 *
 * 索引操作 (合并 gather + scatter)
 * Mechanism: Kernel / Composite
 *
 * @module v7/ops/indexing
 */

import { SchemaT, SchemaShape, SchemaDtype } from '../helpers';
import type { OpEntry } from '../types';

// ============================================================================
// Gather Operations (索引读取)
// ============================================================================

export const indexSelect: OpEntry = {
    name: 'indexSelect',
    mechanism: 'Gather',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '源张量' },
            { name: 'dim', type: SchemaT.Axis(), doc: '选择维度' },
            {
                name: 'index',
                type: SchemaT.Tensor({ dtype: ['int32', 'int64'], ndim: 1 }),
                doc: '1D 索引张量'
            },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('index_select(self.shape, dim, index.shape)'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'index_select',
    doc: '沿维度选择索引: out[...] = self.select(dim, index[...])',
    codegen: { tensorMethod: 'indexSelect' },
};

// ============================================================================
// Scatter Operations (索引写入)
// ============================================================================

export const scatter: OpEntry = {
    name: 'scatter',
    mechanism: 'Scatter',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '目标张量' },
            { name: 'dim', type: SchemaT.Axis(), doc: '散射维度' },
            {
                name: 'index',
                type: SchemaT.Tensor({ dtype: ['int32', 'int64'] }),
                doc: '索引张量 (与 src 相同维度数)'
            },
            { name: 'src', type: SchemaT.Tensor(), doc: '源张量' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'scatter',
    doc: '散射操作: out[index[...]][...] = src[...]',
    codegen: { tensorMethod: 'scatter' },
};

export const scatterAdd: OpEntry = {
    name: 'scatterAdd',
    mechanism: 'Scatter',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '目标张量 (初始值)' },
            { name: 'dim', type: SchemaT.Axis(), doc: '散射维度' },
            {
                name: 'index',
                type: SchemaT.Tensor({ dtype: ['int32', 'int64'] }),
                doc: '索引张量 (与 src 相同维度数)'
            },
            { name: 'src', type: SchemaT.Tensor(), doc: '源张量' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'scatter_add',
    doc: '散射加法: out[index[...]][...] += src[...]',
    codegen: { tensorMethod: 'scatterAdd' },
};

export const scatterReduce: OpEntry = {
    name: 'scatterReduce',
    mechanism: 'Scatter',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '目标张量' },
            { name: 'dim', type: SchemaT.Axis(), doc: '散射维度' },
            {
                name: 'index',
                type: SchemaT.Tensor({ dtype: ['int32', 'int64'] }),
                doc: '索引张量 (与 src 相同维度数)'
            },
            { name: 'src', type: SchemaT.Tensor(), doc: '源张量' },
            {
                name: 'reduce',
                type: SchemaT.String(['sum', 'prod', 'mean', 'amax', 'amin']),
                doc: '归约方式'
            },
            {
                name: 'includeSelf',
                type: SchemaT.Bool(),
                default: true,
                doc: '是否包含 self 原值参与归约'
            },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'scatter_reduce',
    doc: '通用散射归约',
    codegen: { tensorMethod: 'scatterReduce' },
};

// ============================================================================
// Embedding (本质是索引查找)
// ============================================================================

export const embedding: OpEntry = {
    name: 'embedding',
    mechanism: 'Composite',
    signature: {
        params: [
            { name: 'input', type: SchemaT.Tensor({ dtype: ['int32', 'int64'] }), doc: '索引张量' },
            { name: 'weight', type: SchemaT.Tensor({ ndim: 2 }), doc: '嵌入矩阵 (vocab_size, embed_dim)' },
            { name: 'paddingIdx', type: SchemaT.Optional(SchemaT.Scalar('int')), doc: '填充索引 (暂不支持)' },
            { name: 'maxNorm', type: SchemaT.Optional(SchemaT.Scalar('float')), doc: '范数上限 (暂不支持)' },
            { name: 'normType', type: SchemaT.Scalar('float'), default: 2.0, doc: '范数类型 (暂不支持)' },
            { name: 'scaleGradByFreq', type: SchemaT.Bool(), default: false, doc: '按频率缩放梯度 (暂不支持)' },
            { name: 'sparse', type: SchemaT.Bool(), default: false, doc: '稀疏梯度 (暂不支持)' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('embedding(input.shape, weight.shape)'),
    dtype: SchemaDtype.same('weight'),
    dispatchKey: 'embedding',
    doc: '嵌入查找: output[...] = weight[input[...], :]',
    codegen: {
        tensorMethod: false,
        namespace: 'nn.functional'
    },
};
