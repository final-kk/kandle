/**
 * NN-Kit Conv/Pool OpEntry Definitions
 * 
 * Conv 系列算子的 OpSchema 定义，对标 PyTorch。
 * 使用 Kernel 机制，直接分发到后端。
 * 
 * @module opschema/ops/conv
 */

import type { OpEntry } from '../types';
import { SchemaT, SchemaShape, SchemaDtype } from '../helpers';

// ============================================================================
// Conv2d
// ============================================================================

export const conv2d: OpEntry = {
    name: 'conv2d',
    mechanism: 'Window',
    signature: {
        params: [
            { name: 'input', type: SchemaT.Tensor({ ndim: 4 }), doc: '输入 (N, C_in, H, W)' },
            { name: 'weight', type: SchemaT.Tensor({ ndim: 4 }), doc: '卷积核 (C_out, C_in/groups, kH, kW)' },
            { name: 'bias', type: SchemaT.Optional(SchemaT.Tensor({ ndim: 1 })), doc: '偏置 (C_out,)' },
            {
                name: 'stride',
                type: SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int')),
                default: 1,
                doc: '步长 (H, W) 或单值'
            },
            {
                name: 'padding',
                type: SchemaT.Union(
                    SchemaT.Scalar('int'),
                    SchemaT.ScalarList('int'),
                    SchemaT.String(['same', 'valid'])
                ),
                default: 0,
                doc: '填充 (H, W)、单值或 "same"/"valid"'
            },
            {
                name: 'dilation',
                type: SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int')),
                default: 1,
                doc: '膨胀率 (H, W) 或单值'
            },
            {
                name: 'groups',
                type: SchemaT.Scalar('int'),
                default: 1,
                doc: '分组数。groups=1 为标准卷积，groups=in_channels 为深度可分离卷积'
            },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    // 使用 explicit shape rule，形状在 dispatcher 层计算
    shape: SchemaShape.explicit('conv2d_shape(input, weight, stride, padding, dilation)'),
    dtype: SchemaDtype.same('input'),
    dispatchKey: 'conv2d',
    doc: '2D 卷积操作 (PyTorch F.conv2d 兼容)',
    codegen: { namespace: 'nn.functional' },
};

// ============================================================================
// Conv1d
// ============================================================================

export const conv1d: OpEntry = {
    name: 'conv1d',
    mechanism: 'Window',
    signature: {
        params: [
            { name: 'input', type: SchemaT.Tensor({ ndim: 3 }), doc: '输入 (N, C_in, L)' },
            { name: 'weight', type: SchemaT.Tensor({ ndim: 3 }), doc: '卷积核 (C_out, C_in/groups, kL)' },
            { name: 'bias', type: SchemaT.Optional(SchemaT.Tensor({ ndim: 1 })) },
            { name: 'stride', type: SchemaT.Scalar('int'), default: 1 },
            {
                name: 'padding',
                type: SchemaT.Union(SchemaT.Scalar('int'), SchemaT.String(['same', 'valid'])),
                default: 0
            },
            { name: 'dilation', type: SchemaT.Scalar('int'), default: 1 },
            { name: 'groups', type: SchemaT.Scalar('int'), default: 1 },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('conv1d_shape(input, weight, stride, padding, dilation)'),
    dtype: SchemaDtype.same('input'),
    dispatchKey: 'conv1d',
    doc: '1D 卷积操作',
    codegen: { namespace: 'nn.functional' },
};

// ============================================================================
// Conv3d
// ============================================================================

export const conv3d: OpEntry = {
    name: 'conv3d',
    mechanism: 'Window',
    signature: {
        params: [
            { name: 'input', type: SchemaT.Tensor({ ndim: 5 }), doc: '输入 (N, C_in, D, H, W)' },
            { name: 'weight', type: SchemaT.Tensor({ ndim: 5 }), doc: '卷积核 (C_out, C_in/groups, kD, kH, kW)' },
            { name: 'bias', type: SchemaT.Optional(SchemaT.Tensor({ ndim: 1 })) },
            {
                name: 'stride',
                type: SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int')),
                default: 1
            },
            {
                name: 'padding',
                type: SchemaT.Union(
                    SchemaT.Scalar('int'),
                    SchemaT.ScalarList('int'),
                    SchemaT.String(['same', 'valid'])
                ),
                default: 0
            },
            {
                name: 'dilation',
                type: SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int')),
                default: 1
            },
            { name: 'groups', type: SchemaT.Scalar('int'), default: 1 },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('conv3d_shape(input, weight, stride, padding, dilation)'),
    dtype: SchemaDtype.same('input'),
    dispatchKey: 'conv3d',
    doc: '3D 卷积操作',
    codegen: { namespace: 'nn.functional' },
};

// ============================================================================
// ConvTranspose2d (转置卷积/反卷积)
// ============================================================================

export const convTranspose2d: OpEntry = {
    name: 'convTranspose2d',
    mechanism: 'Window',
    signature: {
        params: [
            { name: 'input', type: SchemaT.Tensor({ ndim: 4 }), doc: '输入 (N, C_in, H, W)' },
            { name: 'weight', type: SchemaT.Tensor({ ndim: 4 }), doc: '卷积核 (C_in, C_out/groups, kH, kW)' },
            { name: 'bias', type: SchemaT.Optional(SchemaT.Tensor({ ndim: 1 })) },
            {
                name: 'stride',
                type: SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int')),
                default: 1
            },
            {
                name: 'padding',
                type: SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int')),
                default: 0
            },
            {
                name: 'outputPadding',
                type: SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int')),
                default: 0,
                doc: '输出填充，用于解决 stride > 1 时的形状歧义'
            },
            { name: 'groups', type: SchemaT.Scalar('int'), default: 1 },
            {
                name: 'dilation',
                type: SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int')),
                default: 1
            },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('convTranspose2d_shape(input, weight, stride, padding, outputPadding, dilation)'),
    dtype: SchemaDtype.same('input'),
    dispatchKey: 'conv_transpose2d',
    doc: '2D 转置卷积 (反卷积)',
    codegen: { namespace: 'nn.functional' },
};

// ============================================================================
// MaxPool2d
// ============================================================================

export const maxPool2d: OpEntry = {
    name: 'maxPool2d',
    mechanism: 'Window',
    signature: {
        params: [
            { name: 'input', type: SchemaT.Tensor({ ndim: 4 }), doc: '输入 (N, C, H, W)' },
            {
                name: 'kernelSize',
                type: SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int')),
                doc: '池化窗口大小'
            },
            {
                name: 'stride',
                type: SchemaT.Optional(SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int'))),
                doc: '步长，默认等于 kernelSize'
            },
            {
                name: 'padding',
                type: SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int')),
                default: 0
            },
            {
                name: 'dilation',
                type: SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int')),
                default: 1
            },
            {
                name: 'ceilMode',
                type: SchemaT.Bool(),
                default: false,
                doc: '使用 ceil 而非 floor 计算输出尺寸'
            },
            {
                name: 'returnIndices',
                type: SchemaT.Bool(),
                default: false,
                doc: '是否返回最大值索引'
            },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('pool2d_shape(input, kernelSize, stride, padding, dilation, ceilMode)'),
    dtype: SchemaDtype.same('input'),
    dispatchKey: 'max_pool2d',
    doc: '2D 最大池化',
    codegen: {
        namespace: 'nn.functional',
        conditionalReturn: { param: 'returnIndices', tupleSize: 2 }
    },
};

// ============================================================================
// AvgPool2d
// ============================================================================

export const avgPool2d: OpEntry = {
    name: 'avgPool2d',
    mechanism: 'Window',
    signature: {
        params: [
            { name: 'input', type: SchemaT.Tensor({ ndim: 4 }), doc: '输入 (N, C, H, W)' },
            {
                name: 'kernelSize',
                type: SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int'))
            },
            {
                name: 'stride',
                type: SchemaT.Optional(SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int')))
            },
            {
                name: 'padding',
                type: SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int')),
                default: 0
            },
            {
                name: 'ceilMode',
                type: SchemaT.Bool(),
                default: false
            },
            {
                name: 'countIncludePad',
                type: SchemaT.Bool(),
                default: true,
                doc: '计算平均值时是否包含填充元素'
            },
            {
                name: 'divisorOverride',
                type: SchemaT.Optional(SchemaT.Scalar('int')),
                doc: '指定除数，覆盖默认的池化区域大小'
            },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('pool2d_shape(input, kernelSize, stride, padding, 1, ceilMode)'),
    dtype: SchemaDtype.same('input'),
    dispatchKey: 'avg_pool2d',
    doc: '2D 平均池化',
    codegen: { namespace: 'nn.functional' },
};

// ============================================================================
// MaxPool1d
// ============================================================================

export const maxPool1d: OpEntry = {
    name: 'maxPool1d',
    mechanism: 'Window',
    signature: {
        params: [
            { name: 'input', type: SchemaT.Tensor({ ndim: 3 }), doc: '输入 (N, C, L)' },
            { name: 'kernelSize', type: SchemaT.Scalar('int') },
            { name: 'stride', type: SchemaT.Optional(SchemaT.Scalar('int')) },
            { name: 'padding', type: SchemaT.Scalar('int'), default: 0 },
            { name: 'dilation', type: SchemaT.Scalar('int'), default: 1 },
            { name: 'ceilMode', type: SchemaT.Bool(), default: false },
            { name: 'returnIndices', type: SchemaT.Bool(), default: false },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('pool1d_shape(input, kernelSize, stride, padding, dilation, ceilMode)'),
    dtype: SchemaDtype.same('input'),
    dispatchKey: 'max_pool1d',
    doc: '1D 最大池化',
    codegen: { namespace: 'nn.functional' },
};

// ============================================================================
// AvgPool1d
// ============================================================================

export const avgPool1d: OpEntry = {
    name: 'avgPool1d',
    mechanism: 'Window',
    signature: {
        params: [
            { name: 'input', type: SchemaT.Tensor({ ndim: 3 }), doc: '输入 (N, C, L)' },
            { name: 'kernelSize', type: SchemaT.Scalar('int') },
            { name: 'stride', type: SchemaT.Optional(SchemaT.Scalar('int')) },
            { name: 'padding', type: SchemaT.Scalar('int'), default: 0 },
            { name: 'ceilMode', type: SchemaT.Bool(), default: false },
            { name: 'countIncludePad', type: SchemaT.Bool(), default: true },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('pool1d_shape(input, kernelSize, stride, padding, 1, ceilMode)'),
    dtype: SchemaDtype.same('input'),
    dispatchKey: 'avg_pool1d',
    doc: '1D 平均池化',
    codegen: { namespace: 'nn.functional' },
};

// ============================================================================
// MaxPool3d
// ============================================================================

export const maxPool3d: OpEntry = {
    name: 'maxPool3d',
    mechanism: 'Window',
    signature: {
        params: [
            { name: 'input', type: SchemaT.Tensor({ ndim: 5 }), doc: '输入 (N, C, D, H, W)' },
            {
                name: 'kernelSize',
                type: SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int'))
            },
            {
                name: 'stride',
                type: SchemaT.Optional(SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int')))
            },
            {
                name: 'padding',
                type: SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int')),
                default: 0
            },
            {
                name: 'dilation',
                type: SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int')),
                default: 1
            },
            { name: 'ceilMode', type: SchemaT.Bool(), default: false },
            { name: 'returnIndices', type: SchemaT.Bool(), default: false },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('pool3d_shape(input, kernelSize, stride, padding, dilation, ceilMode)'),
    dtype: SchemaDtype.same('input'),
    dispatchKey: 'max_pool3d',
    doc: '3D 最大池化',
    codegen: { namespace: 'nn.functional' },
};

// ============================================================================
// AvgPool3d
// ============================================================================

export const avgPool3d: OpEntry = {
    name: 'avgPool3d',
    mechanism: 'Window',
    signature: {
        params: [
            { name: 'input', type: SchemaT.Tensor({ ndim: 5 }), doc: '输入 (N, C, D, H, W)' },
            {
                name: 'kernelSize',
                type: SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int'))
            },
            {
                name: 'stride',
                type: SchemaT.Optional(SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int')))
            },
            {
                name: 'padding',
                type: SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int')),
                default: 0
            },
            { name: 'ceilMode', type: SchemaT.Bool(), default: false },
            { name: 'countIncludePad', type: SchemaT.Bool(), default: true },
            { name: 'divisorOverride', type: SchemaT.Optional(SchemaT.Scalar('int')) },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('pool3d_shape(input, kernelSize, stride, padding, 1, ceilMode)'),
    dtype: SchemaDtype.same('input'),
    dispatchKey: 'avg_pool3d',
    doc: '3D 平均池化',
    codegen: { namespace: 'nn.functional' },
};

// ============================================================================
// AdaptiveAvgPool2d (常用于全局平均池化)
// ============================================================================

export const adaptiveAvgPool2d: OpEntry = {
    name: 'adaptiveAvgPool2d',
    mechanism: 'Window',
    signature: {
        params: [
            { name: 'input', type: SchemaT.Tensor({ ndim: 4 }), doc: '输入 (N, C, H, W)' },
            {
                name: 'outputSize',
                type: SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int')),
                doc: '目标输出尺寸 (H_out, W_out)'
            },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('adaptivePool2d_shape(input, outputSize)'),
    dtype: SchemaDtype.same('input'),
    dispatchKey: 'adaptive_avg_pool2d',
    doc: '自适应 2D 平均池化（自动计算 kernel/stride）',
    codegen: { namespace: 'nn.functional' },
};

// ============================================================================
// AdaptiveMaxPool2d
// ============================================================================

export const adaptiveMaxPool2d: OpEntry = {
    name: 'adaptiveMaxPool2d',
    mechanism: 'Window',
    signature: {
        params: [
            { name: 'input', type: SchemaT.Tensor({ ndim: 4 }), doc: '输入 (N, C, H, W)' },
            {
                name: 'outputSize',
                type: SchemaT.Union(SchemaT.Scalar('int'), SchemaT.ScalarList('int'))
            },
            { name: 'returnIndices', type: SchemaT.Bool(), default: false },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('adaptivePool2d_shape(input, outputSize)'),
    dtype: SchemaDtype.same('input'),
    dispatchKey: 'adaptive_max_pool2d',
    doc: '自适应 2D 最大池化',
    codegen: { namespace: 'nn.functional' },
};

// ============================================================================
// 导出所有 Conv/Pool 操作
// ============================================================================

export const convOps = {
    // Conv
    conv1d,
    conv2d,
    conv3d,
    convTranspose2d,

    // MaxPool
    maxPool1d,
    maxPool2d,
    maxPool3d,

    // AvgPool
    avgPool1d,
    avgPool2d,
    avgPool3d,

    // AdaptivePool
    adaptiveAvgPool2d,
    adaptiveMaxPool2d,
};
