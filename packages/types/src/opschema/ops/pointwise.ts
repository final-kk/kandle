/**
 * NN-Kit Operator Schema v7 - Pointwise Operations
 *
 * 逐元素运算 (合并 unary + arithmetic + comparison)
 * Mechanism: Iterator (Map)
 *
 * 包含:
 * - 基础一元: abs, neg, sign
 * - 数学函数: sqrt, exp, log, sin, cos, ...
 * - 基础二元: add, sub, mul, div, pow, ...
 * - 比较运算: eq, lt, gt, ...
 * - 逻辑运算: logicalNot, isnan, isinf, ...
 *
 * @module v7/ops/pointwise
 */

import { SchemaT, SchemaShape, SchemaDtype } from '../helpers';
import type { OpEntry } from '../types';

// ============================================================================
// 基本一元操作
// ============================================================================

export const abs: OpEntry = {
    name: 'abs',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor({ dtype: 'Real' }), doc: '输入张量' },
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
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'abs',
    doc: '逐元素绝对值: |self|',
    codegen: { tensorMethod: 'abs' },
};

export const neg: OpEntry = {
    name: 'neg',
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
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'neg',
    doc: '逐元素取反: -self',
    codegen: { tensorMethod: 'neg' },
};

export const sign: OpEntry = {
    name: 'sign',
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
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'sign',
    doc: '逐元素符号: sign(self)',
    codegen: { tensorMethod: 'sign' },
};

// ============================================================================
// 数学函数
// ============================================================================

export const sqrt: OpEntry = {
    name: 'sqrt',
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
    dispatchKey: 'sqrt',
    doc: '逐元素平方根: sqrt(self)',
    codegen: { tensorMethod: 'sqrt' },
};

export const rsqrt: OpEntry = {
    name: 'rsqrt',
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
    dispatchKey: 'rsqrt',
    doc: '逐元素平方根倒数: 1/sqrt(self)',
    codegen: { tensorMethod: 'rsqrt' },
};

export const square: OpEntry = {
    name: 'square',
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
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'square',
    doc: '逐元素平方: self * self',
    codegen: { tensorMethod: 'square' },
};

export const exp: OpEntry = {
    name: 'exp',
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
    dispatchKey: 'exp',
    doc: '逐元素指数: e^self',
    codegen: { tensorMethod: 'exp' },
};

export const exp2: OpEntry = {
    name: 'exp2',
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
    dispatchKey: 'exp2',
    doc: '逐元素 2 的幂: 2^self',
    codegen: { tensorMethod: 'exp2' },
};

export const expm1: OpEntry = {
    name: 'expm1',
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
    dispatchKey: 'expm1',
    doc: '逐元素 exp(self) - 1',
    codegen: { tensorMethod: 'expm1' },
};

export const log: OpEntry = {
    name: 'log',
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
    dispatchKey: 'log',
    doc: '逐元素自然对数: ln(self)',
    codegen: { tensorMethod: 'log' },
};

export const log2: OpEntry = {
    name: 'log2',
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
    dispatchKey: 'log2',
    doc: '逐元素以 2 为底对数: log2(self)',
    codegen: { tensorMethod: 'log2' },
};

export const log10: OpEntry = {
    name: 'log10',
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
    dispatchKey: 'log10',
    doc: '逐元素以 10 为底对数: log10(self)',
    codegen: { tensorMethod: 'log10' },
};

export const log1p: OpEntry = {
    name: 'log1p',
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
    dispatchKey: 'log1p',
    doc: '逐元素 ln(1 + self)',
    codegen: { tensorMethod: 'log1p' },
};

// ============================================================================
// 三角函数
// ============================================================================

export const sin: OpEntry = {
    name: 'sin',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量 (弧度)' },
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
    dispatchKey: 'sin',
    doc: '逐元素正弦: sin(self)',
    codegen: { tensorMethod: 'sin' },
};

export const cos: OpEntry = {
    name: 'cos',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量 (弧度)' },
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
    dispatchKey: 'cos',
    doc: '逐元素余弦: cos(self)',
    codegen: { tensorMethod: 'cos' },
};

export const tan: OpEntry = {
    name: 'tan',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量 (弧度)' },
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
    dispatchKey: 'tan',
    doc: '逐元素正切: tan(self)',
    codegen: { tensorMethod: 'tan' },
};

export const asin: OpEntry = {
    name: 'asin',
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
    dispatchKey: 'asin',
    doc: '逐元素反正弦: arcsin(self)',
    codegen: { tensorMethod: 'asin' },
};

export const acos: OpEntry = {
    name: 'acos',
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
    dispatchKey: 'acos',
    doc: '逐元素反余弦: arccos(self)',
    codegen: { tensorMethod: 'acos' },
};

export const atan: OpEntry = {
    name: 'atan',
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
    dispatchKey: 'atan',
    doc: '逐元素反正切: arctan(self)',
    codegen: { tensorMethod: 'atan' },
};

/**
 * atan2 - 二参数反正切
 * 
 * 对标 PyTorch: torch.atan2(input, other, *, out=None) -> Tensor
 * 
 * 计算 atan(self/other)，考虑象限返回正确角度。
 * 返回值范围 [-π, π]
 * 
 * @see https://pytorch.org/docs/stable/generated/torch.atan2.html
 */
export const atan2: OpEntry = {
    name: 'atan2',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: 'y 坐标 (分子)' },
            { name: 'other', type: SchemaT.Tensor(), doc: 'x 坐标 (分母)' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'binary',
        tensorInputs: ['self', 'other'],
        scalarArgs: [],
        outputs: ['out'],
    },
    shape: SchemaShape.broadcast('self', 'other'),
    dtype: SchemaDtype.promote(['self', 'other'], 'Floating'),
    dispatchKey: 'atan2',
    doc: '逐元素二参数反正切: atan2(self, other)，返回 [-π, π] 弧度',
    codegen: { tensorMethod: 'atan2' },
};

export const sinh: OpEntry = {
    name: 'sinh',
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
    dispatchKey: 'sinh',
    doc: '逐元素双曲正弦: sinh(self)',
    codegen: { tensorMethod: 'sinh' },
};

export const cosh: OpEntry = {
    name: 'cosh',
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
    dispatchKey: 'cosh',
    doc: '逐元素双曲余弦: cosh(self)',
    codegen: { tensorMethod: 'cosh' },
};

export const tanh: OpEntry = {
    name: 'tanh',
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
    dispatchKey: 'tanh',
    doc: '逐元素双曲正切: tanh(self)',
    codegen: { tensorMethod: 'tanh', namespace: 'nn.functional' },
};



export const asinh: OpEntry = {
    name: 'asinh',
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
    dispatchKey: 'asinh',
    doc: '逐元素反双曲正弦: arcsinh(self)',
    codegen: { tensorMethod: 'asinh' },
};

export const acosh: OpEntry = {
    name: 'acosh',
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
    dispatchKey: 'acosh',
    doc: '逐元素反双曲余弦: arccosh(self)',
    codegen: { tensorMethod: 'acosh' },
};

export const atanh: OpEntry = {
    name: 'atanh',
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
    dispatchKey: 'atanh',
    doc: '逐元素反双曲正切: arctanh(self)',
    codegen: { tensorMethod: 'atanh' },
};

// ============================================================================
// 取整函数
// ============================================================================

export const floor: OpEntry = {
    name: 'floor',
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
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'floor',
    doc: '逐元素向下取整: floor(self)',
    codegen: { tensorMethod: 'floor' },
};

export const ceil: OpEntry = {
    name: 'ceil',
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
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'ceil',
    doc: '逐元素向上取整: ceil(self)',
    codegen: { tensorMethod: 'ceil' },
};

export const round: OpEntry = {
    name: 'round',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'decimals', type: SchemaT.Scalar('int'), default: 0, doc: '小数位数' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['decimals'],
        outputs: ['out'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'round',
    doc: '逐元素四舍五入: round(self)',
    codegen: { tensorMethod: 'round' },
};

export const trunc: OpEntry = {
    name: 'trunc',
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
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'trunc',
    doc: '逐元素向零取整: trunc(self)',
    codegen: { tensorMethod: 'trunc' },
};

export const frac: OpEntry = {
    name: 'frac',
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
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'frac',
    doc: '逐元素取小数部分: frac(self)',
    codegen: { tensorMethod: 'frac' },
};


// ============================================================================
// 逻辑运算
// ============================================================================

export const logicalNot: OpEntry = {
    name: 'logicalNot',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'bool' }) },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: [],
        outputs: ['out'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.fixed('bool'),
    dispatchKey: 'logical_not',
    doc: '逐元素逻辑非: !self',
    codegen: { tensorMethod: 'logicalNot' },
};

// ============================================================================
// 激活相关数学函数 (torch.sigmoid, torch.relu)
// 注意: 这些是 torch 顶级数学函数，不是 nn.Xxx Module
// ============================================================================

export const sigmoid: OpEntry = {
    name: 'sigmoid',
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
    dispatchKey: 'sigmoid',
    doc: '逐元素 Sigmoid: 1 / (1 + exp(-self))',
    codegen: { tensorMethod: 'sigmoid', namespace: 'nn.functional' },
};

export const relu: OpEntry = {
    name: 'relu',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
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
    dispatchKey: 'relu',
    doc: '逐元素 ReLU: max(0, self)',
    codegen: { tensorMethod: 'relu', namespace: 'nn.functional' },
};

// ============================================================================
// 特殊函数
// ============================================================================

export const reciprocal: OpEntry = {
    name: 'reciprocal',
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
    dispatchKey: 'reciprocal',
    doc: '逐元素倒数: 1/self',
    codegen: { tensorMethod: 'reciprocal' },
};

export const erf: OpEntry = {
    name: 'erf',
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
    dispatchKey: 'erf',
    doc: '逐元素误差函数: erf(self)',
    codegen: { tensorMethod: 'erf' },
};

export const erfc: OpEntry = {
    name: 'erfc',
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
    dispatchKey: 'erfc',
    doc: '逐元素互补误差函数: erfc(self)',
    codegen: { tensorMethod: 'erfc' },
};

/**
 * i0 - 0阶修正贝塞尔函数
 * 
 * 对标 PyTorch: torch.i0(input, *, out=None) -> Tensor / torch.special.i0(input)
 * 
 * 计算输入张量每个元素的 0 阶修正贝塞尔函数 I₀(x)
 * 
 * @see https://pytorch.org/docs/stable/generated/torch.special.i0.html
 */
export const i0: OpEntry = {
    name: 'i0',
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
    dispatchKey: 'i0',
    doc: '逐元素计算 0 阶修正贝塞尔函数: I₀(self)',
    codegen: { tensorMethod: 'i0' },
};

/**
 * sinc - 归一化 sinc 函数
 * 
 * 对标 PyTorch: torch.sinc(input, *, out=None) -> Tensor
 * 
 * sinc(x) = sin(πx) / (πx)，当 x=0 时返回 1
 * 
 * @see https://pytorch.org/docs/stable/generated/torch.sinc.html
 */
export const sinc: OpEntry = {
    name: 'sinc',
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
    dispatchKey: 'sinc',
    doc: '逐元素归一化 sinc: sin(πx)/(πx)，x=0 时返回 1',
    codegen: { tensorMethod: 'sinc' },
};

// ============================================================================
// Clamp
// ============================================================================

export const clamp: OpEntry = {
    name: 'clamp',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'min', type: SchemaT.Optional(SchemaT.Scalar()), doc: '最小值' },
            { name: 'max', type: SchemaT.Optional(SchemaT.Scalar()), doc: '最大值' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['min', 'max'],
        outputs: ['out'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'clamp',
    doc: '逐元素截断到 [min, max] 范围',
    codegen: { tensorMethod: 'clamp' },
};

// ============================================================================
// 复数函数
// ============================================================================

export const conj: OpEntry = {
    name: 'conj',
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
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'conj',
    doc: '复数共轭: a+bi -> a-bi',
    codegen: { tensorMethod: 'conj' },
};

export const real: OpEntry = {
    name: 'real',
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
    dispatchKey: 'real',
    doc: '取实部: a+bi -> a',
    codegen: { tensorMethod: 'real' },
};

export const imag: OpEntry = {
    name: 'imag',
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
    dispatchKey: 'imag',
    doc: '取虚部: a+bi -> b',
    codegen: { tensorMethod: 'imag' },
};

/**
 * angle - 复数相位角
 * 
 * 对标 PyTorch: torch.angle(input, *, out=None) -> Tensor
 * 
 * 计算复数张量每个元素的相位角 (弧度)。
 * 对于实数: 正数返回 0，负数返回 π
 * 
 * @see https://pytorch.org/docs/stable/generated/torch.angle.html
 */
export const angle: OpEntry = {
    name: 'angle',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量 (通常为复数)' },
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
    dispatchKey: 'angle',
    doc: '逐元素计算复数相位角 (弧度)，实数返回 0 或 π',
    codegen: { tensorMethod: 'angle' },
};

// 二元算术运算
// ============================================================================

export const add_Tensor: OpEntry = {
    name: 'add',
    variant: 'Tensor',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'other', type: SchemaT.Tensor(), doc: '要加的张量' },
            { name: 'alpha', type: SchemaT.Scalar(), default: 1, doc: 'other 的乘数' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'binary',
        tensorInputs: ['self', 'other'],
        scalarArgs: ['alpha'],
        outputs: ['out'],
    },
    shape: SchemaShape.broadcast('self', 'other'),
    dtype: SchemaDtype.promote(['self', 'other']),
    dispatchKey: 'add',
    doc: '逐元素加法: self + alpha * other',
    codegen: { tensorMethod: 'add' },
};

export const add_Scalar: OpEntry = {
    name: 'add',
    variant: 'Scalar',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'other', type: SchemaT.Scalar(), doc: '要加的数值' },
            { name: 'alpha', type: SchemaT.Scalar(), default: 1, doc: 'other 的乘数' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['other', 'alpha'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'add_scalar',
    doc: '逐元素加法: self + alpha * other (标量版本)',
    codegen: { tensorMethod: 'add' },
};

export const sub_Tensor: OpEntry = {
    name: 'sub',
    variant: 'Tensor',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'other', type: SchemaT.Tensor(), doc: '要减的张量' },
            { name: 'alpha', type: SchemaT.Scalar(), default: 1, doc: 'other 的乘数' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'binary',
        tensorInputs: ['self', 'other'],
        scalarArgs: ['alpha'],
        outputs: ['out'],
    },
    shape: SchemaShape.broadcast('self', 'other'),
    dtype: SchemaDtype.promote(['self', 'other']),
    dispatchKey: 'sub',
    doc: '逐元素减法: self - alpha * other',
    codegen: { tensorMethod: 'sub' },
};

export const sub_Scalar: OpEntry = {
    name: 'sub',
    variant: 'Scalar',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'other', type: SchemaT.Scalar(), doc: '要减的数值' },
            { name: 'alpha', type: SchemaT.Scalar(), default: 1, doc: 'other 的乘数' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['other', 'alpha'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'sub_scalar',
    doc: '逐元素减法: self - alpha * other (标量版本)',
    codegen: { tensorMethod: 'sub' },
};

export const mul_Tensor: OpEntry = {
    name: 'mul',
    variant: 'Tensor',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'other', type: SchemaT.Tensor(), doc: '要乘的张量' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'binary',
        tensorInputs: ['self', 'other'],
        scalarArgs: [],
        outputs: ['out'],
    },
    shape: SchemaShape.broadcast('self', 'other'),
    dtype: SchemaDtype.promote(['self', 'other']),
    dispatchKey: 'mul',
    doc: '逐元素乘法: self * other',
    codegen: { tensorMethod: 'mul' },
};

export const mul_Scalar: OpEntry = {
    name: 'mul',
    variant: 'Scalar',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'other', type: SchemaT.Scalar(), doc: '要乘的数值' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['other'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'mul_scalar',
    doc: '逐元素乘法: self * other (标量版本)',
    codegen: { tensorMethod: 'mul' },
};

export const div_Tensor: OpEntry = {
    name: 'div',
    variant: 'Tensor',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '被除数' },
            { name: 'other', type: SchemaT.Tensor(), doc: '除数' },
            { name: 'roundingMode', type: SchemaT.Optional(SchemaT.String(['trunc', 'floor'])), doc: '取整模式' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'binary',
        tensorInputs: ['self', 'other'],
        scalarArgs: ['roundingMode'],
        outputs: ['out'],
    },
    shape: SchemaShape.broadcast('self', 'other'),
    dtype: SchemaDtype.promote(['self', 'other'], 'Floating'),
    dispatchKey: 'div',
    doc: '逐元素除法: self / other',
    codegen: { tensorMethod: 'div' },
};

export const div_Scalar: OpEntry = {
    name: 'div',
    variant: 'Scalar',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '被除数' },
            { name: 'other', type: SchemaT.Scalar(), doc: '除数' },
            { name: 'roundingMode', type: SchemaT.Optional(SchemaT.String(['trunc', 'floor'])), doc: '取整模式' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['other', 'roundingMode'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.promote(['self'], 'Floating'),
    dispatchKey: 'div_scalar',
    doc: '逐元素除法: self / other (标量版本)',
    codegen: { tensorMethod: 'div' },
};

export const pow_Tensor: OpEntry = {
    name: 'pow',
    variant: 'Tensor',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '底数' },
            { name: 'exponent', type: SchemaT.Tensor(), doc: '指数' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'binary',
        tensorInputs: ['self', 'exponent'],
        scalarArgs: [],
        outputs: ['out'],
    },
    shape: SchemaShape.broadcast('self', 'exponent'),
    dtype: SchemaDtype.promote(['self', 'exponent'], 'Floating'),
    dispatchKey: 'pow',
    doc: '逐元素幂运算: self ^ exponent',
    codegen: { tensorMethod: 'pow' },
};

export const pow_Scalar: OpEntry = {
    name: 'pow',
    variant: 'Scalar',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '底数' },
            { name: 'exponent', type: SchemaT.Scalar(), doc: '指数' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['exponent'],
        outputs: ['out'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.promote(['self'], 'Floating'),
    dispatchKey: 'pow_scalar',
    doc: '逐元素幂运算: self ^ exponent (标量版本)',
    codegen: { tensorMethod: 'pow' },
};

export const fmod_Tensor: OpEntry = {
    name: 'fmod',
    variant: 'Tensor',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '被除数' },
            { name: 'other', type: SchemaT.Tensor(), doc: '除数' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'binary',
        tensorInputs: ['self', 'other'],
        scalarArgs: [],
        outputs: ['out'],
    },
    shape: SchemaShape.broadcast('self', 'other'),
    dtype: SchemaDtype.promote(['self', 'other']),
    dispatchKey: 'fmod',
    doc: 'C++ 风格取模: fmod(self, other)',
    codegen: { tensorMethod: 'fmod' },
};

export const fmod_Scalar: OpEntry = {
    name: 'fmod',
    variant: 'Scalar',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '被除数' },
            { name: 'other', type: SchemaT.Scalar(), doc: '除数' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['other'],
        outputs: ['out'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'fmod_scalar',
    doc: 'C++ 风格取模: fmod(self, other) (标量版本)',
    codegen: { tensorMethod: 'fmod' },
};

export const remainder_Tensor: OpEntry = {
    name: 'remainder',
    variant: 'Tensor',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '被除数' },
            { name: 'other', type: SchemaT.Tensor(), doc: '除数' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'binary',
        tensorInputs: ['self', 'other'],
        scalarArgs: [],
        outputs: ['out'],
    },
    shape: SchemaShape.broadcast('self', 'other'),
    dtype: SchemaDtype.promote(['self', 'other']),
    dispatchKey: 'remainder',
    doc: 'Python 风格取模: remainder(self, other)',
    codegen: { tensorMethod: 'remainder' },
};

export const remainder_Scalar: OpEntry = {
    name: 'remainder',
    variant: 'Scalar',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '被除数' },
            { name: 'other', type: SchemaT.Scalar(), doc: '除数' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['other'],
        outputs: ['out'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'remainder_scalar',
    doc: 'Python 风格取模: remainder(self, other) (标量版本)',
    codegen: { tensorMethod: 'remainder' },
};

export const maximum: OpEntry = {
    name: 'maximum',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '第一个输入张量' },
            { name: 'other', type: SchemaT.Tensor(), doc: '第二个输入张量' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'binary',
        tensorInputs: ['self', 'other'],
        scalarArgs: [],
        outputs: ['out'],
    },
    shape: SchemaShape.broadcast('self', 'other'),
    dtype: SchemaDtype.promote(['self', 'other']),
    dispatchKey: 'maximum',
    doc: '逐元素最大值: max(self, other)',
    codegen: { tensorMethod: 'maximum' },
};

export const minimum: OpEntry = {
    name: 'minimum',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '第一个输入张量' },
            { name: 'other', type: SchemaT.Tensor(), doc: '第二个输入张量' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'binary',
        tensorInputs: ['self', 'other'],
        scalarArgs: [],
        outputs: ['out'],
    },
    shape: SchemaShape.broadcast('self', 'other'),
    dtype: SchemaDtype.promote(['self', 'other']),
    dispatchKey: 'minimum',
    doc: '逐元素最小值: min(self, other)',
    codegen: { tensorMethod: 'minimum' },
};

export const floorDivide_Tensor: OpEntry = {
    name: 'floorDivide',
    variant: 'Tensor',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '被除数' },
            { name: 'other', type: SchemaT.Tensor(), doc: '除数' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'binary',
        tensorInputs: ['self', 'other'],
        scalarArgs: [],
        outputs: ['out'],
    },
    shape: SchemaShape.broadcast('self', 'other'),
    dtype: SchemaDtype.promote(['self', 'other']),
    dispatchKey: 'floor_divide',
    doc: '向下取整除法: floor(self / other)',
    codegen: { tensorMethod: 'floorDivide' },
};

export const floorDivide_Scalar: OpEntry = {
    name: 'floorDivide',
    variant: 'Scalar',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '被除数' },
            { name: 'other', type: SchemaT.Scalar(), doc: '除数' },
            { name: 'out', type: SchemaT.Optional(SchemaT.Tensor()), doc: '输出张量' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['other'],
        outputs: ['out'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.same('self'),
    dispatchKey: 'floor_divide_scalar',
    doc: '向下取整除法: floor(self / other) (标量版本)',
    codegen: { tensorMethod: 'floorDivide' },
};

// ============================================================================
// 比较运算
// ============================================================================

export const eq_Tensor: OpEntry = {
    name: 'eq',
    variant: 'Tensor',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '第一个输入张量' },
            { name: 'other', type: SchemaT.Tensor(), doc: '第二个输入张量' },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'bool' }) },
    },
    iteratorConfig: {
        factory: 'binary',
        tensorInputs: ['self', 'other'],
        scalarArgs: [],
    },
    shape: SchemaShape.broadcast('self', 'other'),
    dtype: SchemaDtype.fixed('bool'),
    dispatchKey: 'eq',
    doc: '逐元素相等比较: self == other',
    codegen: { tensorMethod: 'eq' },
};

export const eq_Scalar: OpEntry = {
    name: 'eq',
    variant: 'Scalar',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'other', type: SchemaT.Scalar(), doc: '要比较的值' },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'bool' }) },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['other'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.fixed('bool'),
    dispatchKey: 'eq_scalar',
    doc: '逐元素相等比较: self == other (标量版本)',
    codegen: { tensorMethod: 'eq' },
};

export const ne_Tensor: OpEntry = {
    name: 'ne',
    variant: 'Tensor',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '第一个输入张量' },
            { name: 'other', type: SchemaT.Tensor(), doc: '第二个输入张量' },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'bool' }) },
    },
    iteratorConfig: {
        factory: 'binary',
        tensorInputs: ['self', 'other'],
        scalarArgs: [],
    },
    shape: SchemaShape.broadcast('self', 'other'),
    dtype: SchemaDtype.fixed('bool'),
    dispatchKey: 'ne',
    doc: '逐元素不等比较: self != other',
    codegen: { tensorMethod: 'ne' },
};

export const ne_Scalar: OpEntry = {
    name: 'ne',
    variant: 'Scalar',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'other', type: SchemaT.Scalar(), doc: '要比较的值' },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'bool' }) },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['other'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.fixed('bool'),
    dispatchKey: 'ne_scalar',
    doc: '逐元素不等比较: self != other (标量版本)',
    codegen: { tensorMethod: 'ne' },
};

export const lt_Tensor: OpEntry = {
    name: 'lt',
    variant: 'Tensor',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '第一个输入张量' },
            { name: 'other', type: SchemaT.Tensor(), doc: '第二个输入张量' },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'bool' }) },
    },
    iteratorConfig: {
        factory: 'binary',
        tensorInputs: ['self', 'other'],
        scalarArgs: [],
    },
    shape: SchemaShape.broadcast('self', 'other'),
    dtype: SchemaDtype.fixed('bool'),
    dispatchKey: 'lt',
    doc: '逐元素小于比较: self < other',
    codegen: { tensorMethod: 'lt' },
};

export const lt_Scalar: OpEntry = {
    name: 'lt',
    variant: 'Scalar',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'other', type: SchemaT.Scalar(), doc: '要比较的值' },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'bool' }) },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['other'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.fixed('bool'),
    dispatchKey: 'lt_scalar',
    doc: '逐元素小于比较: self < other (标量版本)',
    codegen: { tensorMethod: 'lt' },
};

export const le_Tensor: OpEntry = {
    name: 'le',
    variant: 'Tensor',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '第一个输入张量' },
            { name: 'other', type: SchemaT.Tensor(), doc: '第二个输入张量' },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'bool' }) },
    },
    iteratorConfig: {
        factory: 'binary',
        tensorInputs: ['self', 'other'],
        scalarArgs: [],
    },
    shape: SchemaShape.broadcast('self', 'other'),
    dtype: SchemaDtype.fixed('bool'),
    dispatchKey: 'le',
    doc: '逐元素小于等于比较: self <= other',
    codegen: { tensorMethod: 'le' },
};

export const le_Scalar: OpEntry = {
    name: 'le',
    variant: 'Scalar',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'other', type: SchemaT.Scalar(), doc: '要比较的值' },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'bool' }) },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['other'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.fixed('bool'),
    dispatchKey: 'le_scalar',
    doc: '逐元素小于等于比较: self <= other (标量版本)',
    codegen: { tensorMethod: 'le' },
};

export const gt_Tensor: OpEntry = {
    name: 'gt',
    variant: 'Tensor',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '第一个输入张量' },
            { name: 'other', type: SchemaT.Tensor(), doc: '第二个输入张量' },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'bool' }) },
    },
    iteratorConfig: {
        factory: 'binary',
        tensorInputs: ['self', 'other'],
        scalarArgs: [],
    },
    shape: SchemaShape.broadcast('self', 'other'),
    dtype: SchemaDtype.fixed('bool'),
    dispatchKey: 'gt',
    doc: '逐元素大于比较: self > other',
    codegen: { tensorMethod: 'gt' },
};

export const gt_Scalar: OpEntry = {
    name: 'gt',
    variant: 'Scalar',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'other', type: SchemaT.Scalar(), doc: '要比较的值' },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'bool' }) },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['other'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.fixed('bool'),
    dispatchKey: 'gt_scalar',
    doc: '逐元素大于比较: self > other (标量版本)',
    codegen: { tensorMethod: 'gt' },
};

export const ge_Tensor: OpEntry = {
    name: 'ge',
    variant: 'Tensor',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '第一个输入张量' },
            { name: 'other', type: SchemaT.Tensor(), doc: '第二个输入张量' },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'bool' }) },
    },
    iteratorConfig: {
        factory: 'binary',
        tensorInputs: ['self', 'other'],
        scalarArgs: [],
    },
    shape: SchemaShape.broadcast('self', 'other'),
    dtype: SchemaDtype.fixed('bool'),
    dispatchKey: 'ge',
    doc: '逐元素大于等于比较: self >= other',
    codegen: { tensorMethod: 'ge' },
};

export const ge_Scalar: OpEntry = {
    name: 'ge',
    variant: 'Scalar',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
            { name: 'other', type: SchemaT.Scalar(), doc: '要比较的值' },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'bool' }) },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: ['other'],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.fixed('bool'),
    dispatchKey: 'ge_scalar',
    doc: '逐元素大于等于比较: self >= other (标量版本)',
    codegen: { tensorMethod: 'ge' },
};

// ============================================================================
// 特殊检查函数
// ============================================================================

export const isnan: OpEntry = {
    name: 'isnan',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'bool' }) },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: [],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.fixed('bool'),
    dispatchKey: 'isnan',
    doc: '逐元素检查是否为 NaN',
    codegen: { tensorMethod: 'isnan' },
};

export const isinf: OpEntry = {
    name: 'isinf',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'bool' }) },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: [],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.fixed('bool'),
    dispatchKey: 'isinf',
    doc: '逐元素检查是否为无穷大',
    codegen: { tensorMethod: 'isinf' },
};

export const isfinite: OpEntry = {
    name: 'isfinite',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'self', type: SchemaT.Tensor(), doc: '输入张量' },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'bool' }) },
    },
    iteratorConfig: {
        factory: 'unary',
        tensorInputs: ['self'],
        scalarArgs: [],
    },
    shape: SchemaShape.same('self'),
    dtype: SchemaDtype.fixed('bool'),
    dispatchKey: 'isfinite',
    doc: '逐元素检查是否为有限数',
    codegen: { tensorMethod: 'isfinite' },
};

// ============================================================================
// 条件选择
// ============================================================================

export const where: OpEntry = {
    name: 'where',
    mechanism: 'Iterator',
    iteratorType: 'Map',
    signature: {
        params: [
            { name: 'condition', type: SchemaT.Tensor({ dtype: 'bool' }), doc: '条件张量' },
            { name: 'self', type: SchemaT.Tensor(), doc: '条件为真时的值' },
            { name: 'other', type: SchemaT.Tensor(), doc: '条件为假时的值' },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    iteratorConfig: {
        factory: 'ternary',
        tensorInputs: ['condition', 'self', 'other'],
        scalarArgs: [],
    },
    shape: SchemaShape.broadcast('condition', 'self', 'other'),
    dtype: SchemaDtype.promote(['self', 'other']),
    dispatchKey: 'where',
    doc: '根据条件选择: condition ? self : other',
    codegen: { tensorMethod: 'where', thisArg: 'self' },
};
