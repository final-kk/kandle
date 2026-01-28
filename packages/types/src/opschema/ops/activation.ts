/**
 * NN-Kit Operator Schema v7 - Activation Operations
 *
 * 神经网络激活函数
 * - Iterator.Map: gelu, silu, elu, leakyRelu, hardtanh, logsigmoid, selu, dropout
 * - Kernel: softmax, logSoftmax, softmin
 *
 * 注意: sigmoid, relu, tanh 作为数学函数已移至 pointwise.ts
 *
 * @module v7/ops/activation
 */

import { SchemaT, SchemaShape, SchemaDtype } from '../helpers';
import type { OpEntry } from '../types';

// ============================================================================
// 参数化激活函数 (Iterator.Map)
// ============================================================================

export const gelu: OpEntry = {
    name: 'gelu',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'approximate', type: SchemaT.String(['none', 'tanh']), default: 'none' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['approximate'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'gelu',
    codegen: { tensorMethod: 'gelu', namespace: 'nn.functional' },
};

export const silu: OpEntry = {
    name: 'silu',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: [],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'silu',
    codegen: { tensorMethod: 'silu', namespace: 'nn.functional' },
};

export const elu: OpEntry = {
    name: 'elu',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'alpha', type: SchemaT.Scalar(), default: 1.0 },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['alpha'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'elu',
    codegen: { tensorMethod: 'elu', namespace: 'nn.functional' },
};

export const leakyRelu: OpEntry = {
    name: 'leakyRelu',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'negativeSlope', type: SchemaT.Scalar(), default: 0.01 },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['negativeSlope'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'leaky_relu',
    codegen: { tensorMethod: 'leakyRelu', namespace: 'nn.functional' },
};

export const hardtanh: OpEntry = {
    name: 'hardtanh',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'minVal', type: SchemaT.Scalar(), default: -1 },
            { name: 'maxVal', type: SchemaT.Scalar(), default: 1 },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['minVal', 'maxVal'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'hardtanh',
    codegen: { tensorMethod: 'hardtanh', namespace: 'nn.functional' },
};

export const logsigmoid: OpEntry = {
    name: 'logsigmoid',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: [],
        outputs: ['out'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.promote(['self'], 'Floating'),
    dispatchKey: 'logsigmoid',
    doc: 'LogSigmoid 激活: log(1 / (1 + exp(-self)))',
    codegen: { tensorMethod: 'logsigmoid', namespace: 'nn.functional' },
};

export const selu: OpEntry = {
    name: 'selu',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'inplace', type: SchemaT.Bool(), default: false, doc: '原地操作 (暂不支持)' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['inplace'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'selu',
    doc: 'SELU 激活',
    codegen: { tensorMethod: 'selu', namespace: 'nn.functional' },
};

export const dropout: OpEntry = {
    name: 'dropout',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'p', type: SchemaT.Scalar(), default: 0.5 },
            { name: 'training', type: SchemaT.Bool(), default: true },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['p', 'training'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'dropout',
    codegen: { tensorMethod: 'dropout', namespace: 'nn.functional' },
};

// ============================================================================
// 归一化激活函数 (Kernel)
// ============================================================================

export const softmax: OpEntry = {
    name: 'softmax',
    mechanism: 'Normalize',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor({ dtype: 'Floating' }) },
            { name: 'dim', type: SchemaT.Axis(), default: -1 },
            { name: 'dtype', type: SchemaT.Optional(SchemaT.DType()) },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'softmax',
    codegen: { tensorMethod: 'softmax', namespace: 'nn.functional' },
};

export const logSoftmax: OpEntry = {
    name: 'logSoftmax',
    mechanism: 'Normalize',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor({ dtype: 'Floating' }) },
            { name: 'dim', type: SchemaT.Axis(), default: -1 },
            { name: 'dtype', type: SchemaT.Optional(SchemaT.DType()) },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'log_softmax',
    codegen: { tensorMethod: 'logSoftmax', namespace: 'nn.functional' },
};

export const softmin: OpEntry = {
    name: 'softmin',
    mechanism: 'Normalize',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor({ dtype: 'Floating' }), doc: '输入张量' },
            { name: 'dim', type: SchemaT.Axis(), default: -1, doc: '归约维度' },
            { name: 'dtype', type: SchemaT.Optional(SchemaT.DType()), doc: '计算类型' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'softmin',
    doc: 'Softmin 激活: softmax(-self)',
    codegen: { tensorMethod: 'softmin', namespace: 'nn.functional' },
};
