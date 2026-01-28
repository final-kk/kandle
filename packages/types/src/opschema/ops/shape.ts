/**
 * NN-Kit Operator Schema v7 - Shape Operations
 *
 * 形状操作 (纯元数据，零拷贝)
 * Mechanism: View
 *
 * @module v7/ops/shape
 */

import { SchemaT, SchemaShape, SchemaDtype } from '../helpers';
import type { OpEntry } from '../types';

export const reshape: OpEntry = {
    name: 'reshape',
    mechanism: 'View',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'shape', type: SchemaT.Shape() },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('shape'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'reshape',
    codegen: { tensorMethod: 'reshape' },
};

export const view: OpEntry = {
    name: 'view',
    mechanism: 'View',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'shape', type: SchemaT.Shape() },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('shape'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'view',
    codegen: { tensorMethod: 'view' },
};

export const permute: OpEntry = {
    name: 'permute',
    mechanism: 'View',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'dims', type: SchemaT.Axes() },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.permute('self', 'dims'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'permute',
    codegen: { tensorMethod: 'permute' },
};

export const transpose: OpEntry = {
    name: 'transpose',
    mechanism: 'View',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'dim0', type: SchemaT.Axis() },
            { name: 'dim1', type: SchemaT.Axis() },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.transpose('self', 'dim0', 'dim1'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'transpose',
    codegen: { tensorMethod: 'transpose' },
};

export const unsqueeze: OpEntry = {
    name: 'unsqueeze',
    mechanism: 'View',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'dim', type: SchemaT.Axis() },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('unsqueeze(self.shape, dim)'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'unsqueeze',
    codegen: { tensorMethod: 'unsqueeze' },
};

export const squeeze: OpEntry = {
    name: 'squeeze',
    mechanism: 'View',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'dim', type: SchemaT.Optional(SchemaT.Axis()) },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('squeeze(self.shape, dim)'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'squeeze',
    codegen: { tensorMethod: 'squeeze' },
};

export const flatten: OpEntry = {
    name: 'flatten',
    mechanism: 'View',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'startDim', type: SchemaT.Axis(), default: 0 },
            { name: 'endDim', type: SchemaT.Axis(), default: -1 },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('flatten(self.shape, startDim, endDim)'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'flatten',
    codegen: { tensorMethod: 'flatten' },
};

export const expand: OpEntry = {
    name: 'expand',
    mechanism: 'View',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'size', type: SchemaT.Shape() },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('size'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'expand',
    codegen: { tensorMethod: 'expand' },
};

export const select: OpEntry = {
    name: 'select',
    mechanism: 'View',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'dim', type: SchemaT.Axis(), doc: '要选择的维度' },
            { name: 'index', type: SchemaT.Scalar('int'), doc: '要选择的索引 (支持负数)' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('select(self.shape, dim)'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'select',
    doc: '沿指定维度选择单个索引，返回降维的视图',
    codegen: { tensorMethod: 'select' },
};

export const slice: OpEntry = {
    name: 'slice',
    mechanism: 'View',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            {
                name: 'slices',
                type: SchemaT.Union(SchemaT.String(), SchemaT.Scalar('int')),
                doc: 'Python 风格切片字符串（如 "0:2, :"）或整数索引（降维）'
            },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('computed_from_slices'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'slice',
    doc: '张量切片: 使用 Python 风格切片语法或整数索引返回视图',
    codegen: { tensorMethod: 'slice' },
};

/**
 * as_strided - 创建任意 stride 视图
 * 
 * 对标 PyTorch: torch.as_strided(input, size, stride, storage_offset=None) -> Tensor
 * 
 * 创建一个具有指定 size、stride 和 storage_offset 的视图。
 * 这是一个强大但危险的操作，直接操作内存布局。
 * 
 * WARNING: 生成的视图必须只引用原张量存储中的元素，否则会产生运行时错误。
 * 
 * @see https://pytorch.org/docs/stable/generated/torch.as_strided.html
 */
export const asStrided: OpEntry = {
    name: 'asStrided',
    mechanism: 'View',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'size', type: SchemaT.Shape(), doc: '输出张量的形状' },
            { name: 'stride', type: SchemaT.Shape(), doc: '输出张量的步长' },
            {
                name: 'storageOffset',
                type: SchemaT.Optional(SchemaT.Scalar('int')),
                doc: '存储偏移量。默认使用输入张量的偏移'
            },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('size'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'as_strided',
    doc: '创建具有指定 size 和 stride 的视图（STFT 分帧等场景的核心操作）',
    codegen: { tensorMethod: 'asStrided' },
};

// ============================================================================
// Data-copying shape operations (Kernel mechanism)
// ============================================================================

/**
 * cat - 沿指定维度拼接张量序列
 * 
 * 对标 PyTorch: torch.cat(tensors, dim=0, out=None) -> Tensor
 * 
 * 注意: cat 需要数据拷贝，因此使用 Kernel mechanism 而非 View
 * 
 * @see https://pytorch.org/docs/stable/generated/torch.cat.html
 */
export const cat: OpEntry = {
    name: 'cat',
    mechanism: 'Shape',
    signature: {
        params: [
            {
                name: 'tensors',
                type: SchemaT.TensorList(),
                doc: '要拼接的张量序列，除拼接维度外形状必须相同'
            },
            {
                name: 'dim',
                type: SchemaT.Axis(),
                default: 0,
                doc: '拼接的维度 (支持负数索引)'
            },
            {
                name: 'out',
                type: SchemaT.Optional(SchemaT.Tensor()),
                doc: '可选的输出张量'
            },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('concat(tensors.shapes, dim)'),
    dtype: SchemaDtype.promote(['tensors']),
    dispatchKey: 'cat',
    doc: '沿指定维度拼接张量序列',
};

/**
 * stack - 沿新维度堆叠张量序列
 * 
 * 对标 PyTorch: torch.stack(tensors, dim=0, out=None) -> Tensor
 * 
 * 与 cat 不同，stack 会创建一个新的维度
 * 
 * @see https://pytorch.org/docs/stable/generated/torch.stack.html
 */
export const stack: OpEntry = {
    name: 'stack',
    mechanism: 'Shape',
    signature: {
        params: [
            {
                name: 'tensors',
                type: SchemaT.TensorList(),
                doc: '要堆叠的张量序列，所有张量形状必须相同'
            },
            {
                name: 'dim',
                type: SchemaT.Axis(),
                default: 0,
                doc: '插入新维度的位置 (支持负数索引)'
            },
            {
                name: 'out',
                type: SchemaT.Optional(SchemaT.Tensor()),
                doc: '可选的输出张量'
            },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('stack(tensors.shapes, dim)'),
    dtype: SchemaDtype.promote(['tensors']),
    dispatchKey: 'stack',
    doc: '沿新维度堆叠张量序列',
};

// ============================================================================
// repeat_interleave - 元素重复
// ============================================================================

/**
 * repeat_interleave - 沿维度重复张量元素
 *
 * 对标 PyTorch: torch.repeat_interleave(input, repeats, dim=None, *, output_size=None)
 *
 * @see https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html
 *
 * 与 repeat() 的区别:
 * - repeat_interleave: 重复单个元素 [1,2,3] x 2 -> [1,1,2,2,3,3]
 * - repeat: 重复整个张量 [1,2,3] x 2 -> [1,2,3,1,2,3]
 */
export const repeatInterleave: OpEntry = {
    name: 'repeatInterleave',
    mechanism: 'Shape',
    signature: {
        params: [
            {
                name: 'self',
                type: SchemaT.Tensor(),
                doc: '输入张量'
            },
            {
                name: 'repeats',
                type: SchemaT.Union(SchemaT.Scalar('int'), SchemaT.Tensor({ dtype: 'int64' })),
                doc: '每个元素的重复次数（标量或张量）'
            },
            {
                name: 'dim',
                type: SchemaT.Optional(SchemaT.Axis()),
                doc: '沿哪个维度重复。不指定时先展平再重复'
            },
            {
                name: 'outputSize',
                type: SchemaT.Optional(SchemaT.Scalar('int')),
                doc: '可选的输出大小（避免同步计算）'
            },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('repeat_interleave(self.shape, repeats, dim)'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'repeat_interleave',
    doc: '沿维度重复每个元素指定次数',
    codegen: { tensorMethod: 'repeatInterleave' },
};

// ============================================================================
// diff - N阶前向差分
// ============================================================================

/**
 * diff - 计算 N 阶前向差分
 *
 * 对标 PyTorch: torch.diff(input, n=1, dim=-1, prepend=None, append=None)
 *
 * @see https://pytorch.org/docs/stable/generated/torch.diff.html
 *
 * 一阶差分: out[i] = input[i+1] - input[i]
 * 高阶差分: 递归应用一阶差分
 */
export const diff: OpEntry = {
    name: 'diff',
    mechanism: 'Shape',
    signature: {
        params: [
            {
                name: 'self',
                type: SchemaT.Tensor(),
                doc: '输入张量'
            },
            {
                name: 'n',
                type: SchemaT.Scalar('int'),
                default: 1,
                doc: '差分阶数'
            },
            {
                name: 'dim',
                type: SchemaT.Axis(),
                default: -1,
                doc: '差分维度'
            },
            {
                name: 'prepend',
                type: SchemaT.Optional(SchemaT.Tensor()),
                doc: '计算差分前在 dim 维度前置的张量'
            },
            {
                name: 'append',
                type: SchemaT.Optional(SchemaT.Tensor()),
                doc: '计算差分前在 dim 维度后置的张量'
            },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('diff_shape(self, n, dim, prepend, append)'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'diff',
    doc: '计算 N 阶前向差分: out[i] = input[i+1] - input[i]',
    codegen: { tensorMethod: 'diff' },
};

// ============================================================================
// flip - 沿指定维度翻转张量
// ============================================================================

/**
 * flip - 沿给定维度翻转张量
 *
 * 对标 PyTorch: torch.flip(input, dims) -> Tensor
 *
 * @see https://pytorch.org/docs/stable/generated/torch.flip.html
 *
 * 注意: torch.flip 返回数据拷贝 (非视图)，与 numpy.flip 不同
 */
export const flip: OpEntry = {
    name: 'flip',
    mechanism: 'Shape',
    signature: {
        params: [
            {
                name: 'self',
                type: SchemaT.Tensor(),
                doc: '输入张量'
            },
            {
                name: 'dims',
                type: SchemaT.Axes(),
                doc: '要翻转的维度列表 (支持负数索引)'
            },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'flip',
    doc: '沿给定维度翻转张量元素顺序',
    codegen: { tensorMethod: 'flip' },
};

/**
 * fliplr - 左右翻转 (沿 dim=1 翻转)
 *
 * 对标 PyTorch: torch.fliplr(input) -> Tensor
 *
 * @see https://pytorch.org/docs/stable/generated/torch.fliplr.html
 *
 * 要求输入至少 2D
 */
export const fliplr: OpEntry = {
    name: 'fliplr',
    mechanism: 'Shape',
    signature: {
        params: [
            {
                name: 'self',
                type: SchemaT.Tensor(),
                doc: '输入张量 (至少 2D)'
            },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'fliplr',
    doc: '左右翻转 (沿 dim=1 翻转)，要求至少 2D',
    codegen: { tensorMethod: 'fliplr' },
};

/**
 * flipud - 上下翻转 (沿 dim=0 翻转)
 *
 * 对标 PyTorch: torch.flipud(input) -> Tensor
 *
 * @see https://pytorch.org/docs/stable/generated/torch.flipud.html
 *
 * 要求输入至少 1D
 */
export const flipud: OpEntry = {
    name: 'flipud',
    mechanism: 'Shape',
    signature: {
        params: [
            {
                name: 'self',
                type: SchemaT.Tensor(),
                doc: '输入张量 (至少 1D)'
            },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'flipud',
    doc: '上下翻转 (沿 dim=0 翻转)，要求至少 1D',
    codegen: { tensorMethod: 'flipud' },
};
