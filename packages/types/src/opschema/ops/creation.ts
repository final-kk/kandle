/**
 * NN-Kit Operator Schema v7 - Creation Operations
 *
 * 创建新张量的操作
 * Mechanism: Factory
 *
 * @module v7/ops/creation
 */

import { SchemaT, SchemaShape, SchemaDtype } from '../helpers';
import type { OpEntry } from '../types';

export const zeros: OpEntry = {
    name: 'zeros',
    mechanism: 'Factory',
    signature: {
        params: [
            { name: 'size', type: SchemaT.Shape() },
            { name: 'dtype', type: SchemaT.DType(), default: 'float32' },
            { name: 'device', type: SchemaT.Optional(SchemaT.Device()) },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('size'),
    dtype: SchemaDtype.explicit('dtype'),
    dispatchKey: 'zeros',
    codegen: { tensorMethod: false, staticMethod: true },
};

export const ones: OpEntry = {
    name: 'ones',
    mechanism: 'Factory',
    signature: {
        params: [
            { name: 'size', type: SchemaT.Shape() },
            { name: 'dtype', type: SchemaT.DType(), default: 'float32' },
            { name: 'device', type: SchemaT.Optional(SchemaT.Device()) },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('size'),
    dtype: SchemaDtype.explicit('dtype'),
    dispatchKey: 'ones',
    codegen: { tensorMethod: false, staticMethod: true },
};

export const empty: OpEntry = {
    name: 'empty',
    mechanism: 'Factory',
    signature: {
        params: [
            { name: 'size', type: SchemaT.Shape() },
            { name: 'dtype', type: SchemaT.DType(), default: 'float32' },
            { name: 'device', type: SchemaT.Optional(SchemaT.Device()) },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('size'),
    dtype: SchemaDtype.explicit('dtype'),
    dispatchKey: 'empty',
    codegen: { tensorMethod: false, staticMethod: true },
};

export const full: OpEntry = {
    name: 'full',
    mechanism: 'Factory',
    signature: {
        params: [
            { name: 'size', type: SchemaT.Shape() },
            { name: 'fillValue', type: SchemaT.Scalar() },
            { name: 'dtype', type: SchemaT.Optional(SchemaT.DType()) },
            { name: 'device', type: SchemaT.Optional(SchemaT.Device()) },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('size'),
    dtype: SchemaDtype.explicit('dtype', 'fillValue'),
    dispatchKey: 'full',
    codegen: { tensorMethod: false, staticMethod: true },
};

export const zerosLike: OpEntry = {
    name: 'zerosLike',
    mechanism: 'Factory',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'dtype', type: SchemaT.Optional(SchemaT.DType()) },
            { name: 'device', type: SchemaT.Optional(SchemaT.Device()) },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.explicit('dtype', 'self'),
    dispatchKey: 'zeros_like',
    codegen: { tensorMethod: 'zerosLike' },
};

export const onesLike: OpEntry = {
    name: 'onesLike',
    mechanism: 'Factory',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'dtype', type: SchemaT.Optional(SchemaT.DType()) },
            { name: 'device', type: SchemaT.Optional(SchemaT.Device()) },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.explicit('dtype', 'self'),
    dispatchKey: 'ones_like',
    codegen: { tensorMethod: 'onesLike' },
};

export const emptyLike: OpEntry = {
    name: 'emptyLike',
    mechanism: 'Factory',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor() },
            { name: 'dtype', type: SchemaT.Optional(SchemaT.DType()) },
            { name: 'device', type: SchemaT.Optional(SchemaT.Device()) },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.explicit('dtype', 'self'),
    dispatchKey: 'empty_like',
    codegen: { tensorMethod: 'emptyLike' },
};

export const arange: OpEntry = {
    name: 'arange',
    mechanism: 'Factory',
    signature: {
        params: [
            { name: 'start', type: SchemaT.Scalar() },
            { name: 'end', type: SchemaT.Optional(SchemaT.Scalar()) },
            { name: 'step', type: SchemaT.Scalar(), default: 1 },
            { name: 'dtype', type: SchemaT.Optional(SchemaT.DType()) },
            { name: 'device', type: SchemaT.Optional(SchemaT.Device()) },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('[ceil((end - start) / step)]'),
    dtype: SchemaDtype.explicit('dtype'),
    dispatchKey: 'arange',
    codegen: { tensorMethod: false, staticMethod: true },
};

export const linspace: OpEntry = {
    name: 'linspace',
    mechanism: 'Factory',
    signature: {
        params: [
            { name: 'start', type: SchemaT.Scalar() },
            { name: 'end', type: SchemaT.Scalar() },
            { name: 'steps', type: SchemaT.Scalar('int') },
            { name: 'dtype', type: SchemaT.Optional(SchemaT.DType()) },
            { name: 'device', type: SchemaT.Optional(SchemaT.Device()) },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('[steps]'),
    dtype: SchemaDtype.explicit('dtype'),
    dispatchKey: 'linspace',
    codegen: { tensorMethod: false, staticMethod: true },
};

export const rand: OpEntry = {
    name: 'rand',
    mechanism: 'Factory',
    signature: {
        params: [
            { name: 'size', type: SchemaT.Shape() },
            { name: 'dtype', type: SchemaT.DType(), default: 'float32' },
            { name: 'device', type: SchemaT.Optional(SchemaT.Device()) },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('size'),
    dtype: SchemaDtype.explicit('dtype'),
    dispatchKey: 'rand',
    codegen: { tensorMethod: false, staticMethod: true },
};

export const randn: OpEntry = {
    name: 'randn',
    mechanism: 'Factory',
    signature: {
        params: [
            { name: 'size', type: SchemaT.Shape() },
            { name: 'dtype', type: SchemaT.DType(), default: 'float32' },
            { name: 'device', type: SchemaT.Optional(SchemaT.Device()) },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('size'),
    dtype: SchemaDtype.explicit('dtype'),
    dispatchKey: 'randn',
    codegen: { tensorMethod: false, staticMethod: true },
};

export const randint: OpEntry = {
    name: 'randint',
    mechanism: 'Factory',
    signature: {
        params: [
            { name: 'low', type: SchemaT.Scalar('int') },
            { name: 'high', type: SchemaT.Scalar('int') },
            { name: 'size', type: SchemaT.Shape() },
            { name: 'dtype', type: SchemaT.DType(), default: 'int64' },
            { name: 'device', type: SchemaT.Optional(SchemaT.Device()) },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('size'),
    dtype: SchemaDtype.explicit('dtype'),
    dispatchKey: 'randint',
    codegen: { tensorMethod: false, staticMethod: true },
};

export const eye: OpEntry = {
    name: 'eye',
    mechanism: 'Factory',
    signature: {
        params: [
            { name: 'n', type: SchemaT.Scalar('int') },
            { name: 'm', type: SchemaT.Optional(SchemaT.Scalar('int')) },
            { name: 'dtype', type: SchemaT.DType(), default: 'float32' },
            { name: 'device', type: SchemaT.Optional(SchemaT.Device()) },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('[n, m ?? n]'),
    dtype: SchemaDtype.explicit('dtype'),
    dispatchKey: 'eye',
    codegen: { tensorMethod: false, staticMethod: true },
};

// ============================================================================
// pad - N 维填充
// ============================================================================

/**
 * pad - N 维张量填充
 * 
 * 对标 PyTorch: torch.nn.functional.pad(input, pad, mode='constant', value=None)
 * 
 * @see https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
 * 
 * pad 参数格式: [padding_left, padding_right, padding_top, padding_bottom, ...]
 * 从最后一个维度开始往前填充
 */
export const pad: OpEntry = {
    name: 'pad',
    mechanism: 'Factory',
    signature: {
        params: [
            { name: 'input', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'pad', type: SchemaT.Shape(), doc: '填充大小 [left, right, top, bottom, ...]，从最后一维往前' },
            { name: 'mode', type: { kind: 'Optional', inner: { kind: 'String', oneOf: ['constant', 'reflect', 'replicate', 'circular'] } }, doc: '填充模式。默认: constant' },
            { name: 'value', type: SchemaT.Optional(SchemaT.Scalar('float')), doc: 'constant 模式的填充值。默认: 0' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('padded_shape(input.shape, pad)'),
    dtype: SchemaDtype.same('input'),
    dispatchKey: 'pad',
    doc: 'N 维张量填充 (STFT center padding 等场景)',
    codegen: { tensorMethod: false, staticMethod: false },
};

// ============================================================================
// multinomial - 多项式分布采样
// ============================================================================

/**
 * multinomial - 从多项式概率分布中采样索引
 *
 * 对标 PyTorch: torch.multinomial(input, num_samples, replacement=False, *, generator=None, out=None)
 *
 * @see https://pytorch.org/docs/stable/generated/torch.multinomial.html
 *
 * 注意事项:
 * - input 不需要归一化（会自动归一化为概率）
 * - input 必须非负，且每行至少有一个非零元素
 * - 当 replacement=false 时，num_samples 不能超过非零元素数量
 */
export const multinomial: OpEntry = {
    name: 'multinomial',
    mechanism: 'Factory',
    signature: {
        params: [
            {
                name: 'input',
                type: SchemaT.Tensor({ dtype: 'Floating' }),
                doc: '概率/权重张量 (1D或2D)，值必须非负'
            },
            {
                name: 'numSamples',
                type: SchemaT.Scalar('int'),
                doc: '要采样的数量'
            },
            {
                name: 'replacement',
                type: SchemaT.Bool(),
                default: false,
                doc: '是否有放回采样'
            },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'int64' }) },
    },
    shape: SchemaShape.explicit('multinomial(input.shape, numSamples)'),
    dtype: SchemaDtype.fixed('int64'),
    dispatchKey: 'multinomial',
    doc: '从多项式分布中采样索引',
    codegen: { tensorMethod: 'multinomial', staticMethod: true },
};

// ============================================================================
// Window Functions - 窗函数生成 (信号处理)
// ============================================================================

/**
 * hann_window - 生成 Hann 窗函数
 * 
 * 对标 PyTorch: torch.hann_window(window_length, periodic=True, *, dtype=None, device=None)
 * 
 * @see https://pytorch.org/docs/stable/generated/torch.hann_window.html
 */
export const hannWindow: OpEntry = {
    name: 'hannWindow',
    mechanism: 'WindowFunc',
    signature: {
        params: [
            { name: 'windowLength', type: SchemaT.Scalar('int'), doc: '窗口长度' },
            { name: 'periodic', type: SchemaT.Bool(), default: true, doc: '是否为周期窗口 (用于 STFT)' },
            { name: 'dtype', type: SchemaT.DType(), default: 'float32' },
            { name: 'device', type: SchemaT.Optional(SchemaT.Device()) },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'Floating', ndim: 1 }) },
    },
    shape: SchemaShape.explicit('[windowLength]'),
    dtype: SchemaDtype.explicit('dtype'),
    dispatchKey: 'windowfunc.hann',
    doc: '生成 Hann 窗函数',
    codegen: { tensorMethod: false, staticMethod: true },
};

/**
 * hamming_window - 生成 Hamming 窗函数
 * 
 * 对标 PyTorch: torch.hamming_window(window_length, periodic=True, alpha=0.54, beta=0.46, *, dtype=None, device=None)
 * 
 * @see https://pytorch.org/docs/stable/generated/torch.hamming_window.html
 */
export const hammingWindow: OpEntry = {
    name: 'hammingWindow',
    mechanism: 'WindowFunc',
    signature: {
        params: [
            { name: 'windowLength', type: SchemaT.Scalar('int'), doc: '窗口长度' },
            { name: 'periodic', type: SchemaT.Bool(), default: true, doc: '是否为周期窗口' },
            { name: 'alpha', type: SchemaT.Scalar('float'), default: 0.54, doc: 'Hamming 窗口系数 α' },
            { name: 'beta', type: SchemaT.Scalar('float'), default: 0.46, doc: 'Hamming 窗口系数 β' },
            { name: 'dtype', type: SchemaT.DType(), default: 'float32' },
            { name: 'device', type: SchemaT.Optional(SchemaT.Device()) },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'Floating', ndim: 1 }) },
    },
    shape: SchemaShape.explicit('[windowLength]'),
    dtype: SchemaDtype.explicit('dtype'),
    dispatchKey: 'windowfunc.hamming',
    doc: '生成 Hamming 窗函数',
    codegen: { tensorMethod: false, staticMethod: true },
};

/**
 * blackman_window - 生成 Blackman 窗函数
 * 
 * 对标 PyTorch: torch.blackman_window(window_length, periodic=True, *, dtype=None, device=None)
 * 
 * @see https://pytorch.org/docs/stable/generated/torch.blackman_window.html
 */
export const blackmanWindow: OpEntry = {
    name: 'blackmanWindow',
    mechanism: 'WindowFunc',
    signature: {
        params: [
            { name: 'windowLength', type: SchemaT.Scalar('int'), doc: '窗口长度' },
            { name: 'periodic', type: SchemaT.Bool(), default: true, doc: '是否为周期窗口' },
            { name: 'dtype', type: SchemaT.DType(), default: 'float32' },
            { name: 'device', type: SchemaT.Optional(SchemaT.Device()) },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'Floating', ndim: 1 }) },
    },
    shape: SchemaShape.explicit('[windowLength]'),
    dtype: SchemaDtype.explicit('dtype'),
    dispatchKey: 'windowfunc.blackman',
    doc: '生成 Blackman 窗函数',
    codegen: { tensorMethod: false, staticMethod: true },
};

/**
 * bartlett_window - 生成 Bartlett 窗函数
 * 
 * 对标 PyTorch: torch.bartlett_window(window_length, periodic=True, *, dtype=None, device=None)
 * 
 * @see https://pytorch.org/docs/stable/generated/torch.bartlett_window.html
 */
export const bartlettWindow: OpEntry = {
    name: 'bartlettWindow',
    mechanism: 'WindowFunc',
    signature: {
        params: [
            { name: 'windowLength', type: SchemaT.Scalar('int'), doc: '窗口长度' },
            { name: 'periodic', type: SchemaT.Bool(), default: true, doc: '是否为周期窗口' },
            { name: 'dtype', type: SchemaT.DType(), default: 'float32' },
            { name: 'device', type: SchemaT.Optional(SchemaT.Device()) },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'Floating', ndim: 1 }) },
    },
    shape: SchemaShape.explicit('[windowLength]'),
    dtype: SchemaDtype.explicit('dtype'),
    dispatchKey: 'windowfunc.bartlett',
    doc: '生成 Bartlett 窗函数',
    codegen: { tensorMethod: false, staticMethod: true },
};

/**
 * kaiser_window - 生成 Kaiser 窗函数
 * 
 * 对标 PyTorch: torch.kaiser_window(window_length, periodic=True, beta=12.0, *, dtype=None, device=None)
 * 
 * @see https://pytorch.org/docs/stable/generated/torch.kaiser_window.html
 */
export const kaiserWindow: OpEntry = {
    name: 'kaiserWindow',
    mechanism: 'WindowFunc',
    signature: {
        params: [
            { name: 'windowLength', type: SchemaT.Scalar('int'), doc: '窗口长度' },
            { name: 'periodic', type: SchemaT.Bool(), default: true, doc: '是否为周期窗口' },
            { name: 'beta', type: SchemaT.Scalar('float'), default: 12.0, doc: 'Kaiser 窗口形状参数' },
            { name: 'dtype', type: SchemaT.DType(), default: 'float32' },
            { name: 'device', type: SchemaT.Optional(SchemaT.Device()) },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'Floating', ndim: 1 }) },
    },
    shape: SchemaShape.explicit('[windowLength]'),
    dtype: SchemaDtype.explicit('dtype'),
    dispatchKey: 'windowfunc.kaiser',
    doc: '生成 Kaiser 窗函数 (需要 Bessel I₀ 函数)',
    codegen: { tensorMethod: false, staticMethod: true },
};

