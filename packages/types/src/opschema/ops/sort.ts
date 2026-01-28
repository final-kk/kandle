/**
 * NN-Kit Operator Schema v6 - Sort Operations
 *
 * 排序操作
 * Mechanism: Kernel
 *
 * @module v6/ops/sort
 */

import { SchemaT, SchemaShape, SchemaDtype } from '../helpers';
import type { OpEntry } from '../types';

export const sort: OpEntry = {
    name: 'sort',
    mechanism: 'Sort',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'dim', type: SchemaT.Axis(), default: -1 },
            { name: 'descending', type: SchemaT.Bool(), default: false },
            { name: 'stable', type: SchemaT.Bool(), default: false },
        ],
        returns: {
            tuple: [
                { name: 'values', type: SchemaT.Tensor() },
                { name: 'indices', type: SchemaT.Tensor({ dtype: 'int64' }) },
            ],
        },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'sort',
    codegen: { tensorMethod: 'sort' },
};

export const argsort: OpEntry = {
    name: 'argsort',
    mechanism: 'Sort',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'dim', type: SchemaT.Axis(), default: -1 },
            { name: 'descending', type: SchemaT.Bool(), default: false },
            { name: 'stable', type: SchemaT.Bool(), default: false },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'int64' }) },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.fixed('int64'),
    dispatchKey: 'argsort',
    codegen: { tensorMethod: 'argsort' },
};

export const topk: OpEntry = {
    name: 'topk',
    mechanism: 'Sort',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'k', type: SchemaT.Scalar('int') },
            { name: 'dim', type: SchemaT.Axis(), default: -1 },
            { name: 'largest', type: SchemaT.Bool(), default: true },
            { name: 'sorted', type: SchemaT.Bool(), default: true },
        ],
        returns: {
            tuple: [
                { name: 'values', type: SchemaT.Tensor() },
                { name: 'indices', type: SchemaT.Tensor({ dtype: 'int64' }) },
            ],
        },
    },
    shape: SchemaShape.explicit('topk(self.shape, k, dim)'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'topk',
    codegen: { tensorMethod: 'topk' },
};
