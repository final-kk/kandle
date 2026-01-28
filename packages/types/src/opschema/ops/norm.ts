/**
 * NN-Kit Operator Schema v7 - Normalization Operations
 *
 * 归一化操作
 * Mechanism: Kernel
 *
 * @module v7/ops/norm
 */

import { SchemaT, SchemaShape, SchemaDtype } from '../helpers';
import type { OpEntry } from '../types';

export const batchNorm: OpEntry = {
    name: 'batchNorm',
    mechanism: 'Normalize',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'runningMean', type: SchemaT.Optional(SchemaT.Tensor()), doc: '运行时均值' },
            { name: 'runningVar', type: SchemaT.Optional(SchemaT.Tensor()), doc: '运行时方差' },
            { name: 'weight', type: SchemaT.Optional(SchemaT.Tensor()), doc: '权重 (gamma)' },
            { name: 'bias', type: SchemaT.Optional(SchemaT.Tensor()), doc: '偏置 (beta)' },
            { name: 'training', type: SchemaT.Bool(), default: false, doc: '训练模式' },
            { name: 'momentum', type: SchemaT.Scalar(), default: 0.1, doc: '动量' },
            { name: 'eps', type: SchemaT.Scalar(), default: 1e-5, doc: '数值稳定项' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'batch_norm',
    doc: 'Batch Normalization',
    codegen: { tensorMethod: 'batchNorm', namespace: 'nn.functional' },
};

export const groupNorm: OpEntry = {
    name: 'groupNorm',
    mechanism: 'Normalize',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'numGroups', type: SchemaT.Scalar('int'), doc: '分组数' },
            { name: 'weight', type: SchemaT.Optional(SchemaT.Tensor()), doc: '权重' },
            { name: 'bias', type: SchemaT.Optional(SchemaT.Tensor()), doc: '偏置' },
            { name: 'eps', type: SchemaT.Scalar(), default: 1e-5, doc: '数值稳定项' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'group_norm',
    doc: 'Group Normalization',
    codegen: { tensorMethod: 'groupNorm', namespace: 'nn.functional' },
};

export const layerNorm: OpEntry = {
    name: 'layerNorm',
    mechanism: 'Normalize',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'normalizedShape', type: SchemaT.Shape(), doc: '归一化形状' },
            { name: 'weight', type: SchemaT.Optional(SchemaT.Tensor()), doc: '权重' },
            { name: 'bias', type: SchemaT.Optional(SchemaT.Tensor()), doc: '偏置' },
            { name: 'eps', type: SchemaT.Scalar(), default: 1e-5, doc: '数值稳定项' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'layer_norm',
    doc: 'Layer Normalization',
    codegen: { tensorMethod: 'layerNorm', namespace: 'nn.functional' },
};

export const rmsNorm: OpEntry = {
    name: 'rmsNorm',
    mechanism: 'Normalize',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'normalizedShape', type: SchemaT.Shape(), doc: '归一化形状' },
            { name: 'weight', type: SchemaT.Optional(SchemaT.Tensor()), doc: '权重' },
            { name: 'eps', type: SchemaT.Scalar(), default: 1e-5, doc: '数值稳定项' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'rms_norm',
    doc: 'RMS Normalization',
    codegen: { tensorMethod: 'rmsNorm', namespace: 'nn.functional' },
};

export const normalize: OpEntry = {
    name: 'normalize',
    mechanism: 'Normalize',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'p', type: SchemaT.Scalar(), default: 2.0, doc: '范数指数' },
            { name: 'dim', type: SchemaT.Axis(), default: 1, doc: '维度' },
            { name: 'eps', type: SchemaT.Scalar(), default: 1e-12, doc: '数值稳定项' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.promote(['self'], 'Floating'),
    dispatchKey: 'lp_normalize',
    doc: 'F.normalize: self / self.norm()',
    codegen: { tensorMethod: 'normalize', namespace: 'nn.functional' },
};
