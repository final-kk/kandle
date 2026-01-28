/**
 * NN-Kit Operator Schema v7 - Memory Operations
 *
 * 内存操作
 * Mechanism: Copy
 *
 * @module v7/ops/memory
 */

import { SchemaT, SchemaShape, SchemaDtype } from '../helpers';
import type { OpEntry } from '../types';

export const contiguous: OpEntry = {
    name: 'contiguous',
    mechanism: 'Copy',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            {
                name: 'memoryFormat',
                type: SchemaT.Optional(SchemaT.MemoryFormat()),
                default: 'contiguous',
                doc: '目标内存格式: "contiguous" (NCHW) 或 "channels_last" (NHWC)'
            },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'contiguous',
    doc: '确保 tensor 按指定格式连续存储。如果已是目标格式则返回原 tensor。',
    codegen: { tensorMethod: 'contiguous' },
};

export const clone: OpEntry = {
    name: 'clone',
    mechanism: 'Copy',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'clone',
    codegen: { tensorMethod: 'clone' },
};

export const cast: OpEntry = {
    name: 'cast',
    mechanism: 'Copy',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'dtype', type: SchemaT.DType() },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.explicit('dtype'),
    dispatchKey: 'cast',
    doc: '类型转换: 将 tensor 转换为指定的 dtype',
    codegen: { tensorMethod: 'cast' },
};

export const to: OpEntry = {
    name: 'to',
    mechanism: 'Copy',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'dtype', type: SchemaT.Optional(SchemaT.DType()) },
            { name: 'device', type: SchemaT.Optional(SchemaT.Device()) },
            { name: 'copy', type: SchemaT.Bool(), default: false },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.explicit('dtype', 'self'),
    dispatchKey: 'to',
    codegen: { tensorMethod: 'to' },
};

/**
 * copy_ - 原地拷贝 (In-place copy)
 *
 * 将 src 的内容拷贝到 self，self 必须与 src 形状广播兼容。
 * 这是 PyTorch 的标准 in-place 操作，常用于写入 view。
 *
 * @example
 * ```ts
 * // 写入 slice view
 * const view = slice(cache, ':, :, 100:105, :');
 * copy_(view, newData);  // 原地写入
 * ```
 */
export const copy_: OpEntry = {
    name: 'copy_',
    mechanism: 'Copy',
    signature: {
        params: [
            {
                name: 'self',
                type: SchemaT.Tensor(),
                doc: '目标张量 (会被原地修改)'
            },
            {
                name: 'src',
                type: SchemaT.Tensor(),
                doc: '源张量'
            },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'copy_',
    doc: '原地拷贝: 将 src 拷贝到 self (in-place)',
    codegen: {
        tensorMethod: 'copy_',
    },
};
