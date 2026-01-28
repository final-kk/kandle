/**
 * NN-Kit Operator Schema v6 - Scan Operations
 *
 * 扫描操作 (累积操作)
 * Mechanism: Iterator (Scan)
 *
 * @module v6/ops/scan
 */

import { SchemaT, SchemaShape, SchemaDtype } from '../helpers';
import type { OpEntry } from '../types';

export const cumsum: OpEntry = {
    name: 'cumsum',
    mechanism: 'Iterator',
    iteratorType: 'Scan',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'dim', type: SchemaT.Axis() },
            { name: 'dtype', type: SchemaT.Optional(SchemaT.DType()) },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'scan',
        tensorInputs: ['self'],
        scalarArgs: ['dim', 'dtype'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.promote(['self'], 'SignedIntegral'),
    dispatchKey: 'cumsum',
    codegen: { tensorMethod: 'cumsum' },
};

export const cumprod: OpEntry = {
    name: 'cumprod',
    mechanism: 'Iterator',
    iteratorType: 'Scan',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'dim', type: SchemaT.Axis() },
            { name: 'dtype', type: SchemaT.Optional(SchemaT.DType()) },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'scan',
        tensorInputs: ['self'],
        scalarArgs: ['dim', 'dtype'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.promote(['self'], 'SignedIntegral'),
    dispatchKey: 'cumprod',
    codegen: { tensorMethod: 'cumprod' },
};

export const cummax: OpEntry = {
    name: 'cummax',
    mechanism: 'Iterator',
    iteratorType: 'Scan',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'dim', type: SchemaT.Axis() },
        ],
        returns: {
            tuple: [
                { name: 'values', type: SchemaT.Tensor() },
                { name: 'indices', type: SchemaT.Tensor({ dtype: 'int64' }) },
            ],
        },
    },
    iteratorConfig: {
        factory: 'scan',
        tensorInputs: ['self'],
        scalarArgs: ['dim'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'cummax',
    codegen: { tensorMethod: 'cummax' },
};

export const cummin: OpEntry = {
    name: 'cummin',
    mechanism: 'Iterator',
    iteratorType: 'Scan',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'dim', type: SchemaT.Axis() },
        ],
        returns: {
            tuple: [
                { name: 'values', type: SchemaT.Tensor() },
                { name: 'indices', type: SchemaT.Tensor({ dtype: 'int64' }) },
            ],
        },
    },
    iteratorConfig: {
        factory: 'scan',
        tensorInputs: ['self'],
        scalarArgs: ['dim'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'cummin',
    codegen: { tensorMethod: 'cummin' },
};
