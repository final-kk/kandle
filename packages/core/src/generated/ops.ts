/**
 * v5 Generated Operators
 * DO NOT EDIT - Generated from OpRegistry
 *
 * Each operator uses typeof branching to dispatch to the correct variant
 */

import type { ITensorHandle, DType } from '@kandle/types';
import { Tensor } from '../tensor';
import * as internal from './internal';

// ============================================================================
// Operators
// ============================================================================

/** 逐元素绝对值: |self| */
export function abs(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.abs(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素反余弦: arccos(self) */
export function acos(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.acos(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素反双曲余弦: arccosh(self) */
export function acosh(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.acosh(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素加法: self + alpha * other (标量版本) (Scalar variant) */
export function add(self: Tensor, other: number, alpha?: number): Tensor;
/** 逐元素加法: self + alpha * other (Tensor variant) */
export function add(self: Tensor, other: Tensor, alpha?: number, out?: Tensor): Tensor;
export function add(
    self: Tensor,
    other: number | Tensor,
    alpha?: number,
    out?: Tensor
): Tensor | [Tensor, Tensor] {
    if (typeof other === 'number') {
        const result = internal.add_Scalar(self._handle, other, alpha);
        return new Tensor(result);
    } else {
        const result = internal.add_Tensor(self._handle, (other as Tensor)._handle, alpha, out?._handle);
        return new Tensor(result);
    }
}

/** 逐元素计算复数相位角 (弧度)，实数返回 0 或 π */
export function angle(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.angle(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素反正弦: arcsin(self) */
export function asin(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.asin(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素反双曲正弦: arcsinh(self) */
export function asinh(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.asinh(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素反正切: arctan(self) */
export function atan(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.atan(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素二参数反正切: atan2(self, other)，返回 [-π, π] 弧度 */
export function atan2(
    self: Tensor,
    other: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.atan2(self._handle, other._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素反双曲正切: arctanh(self) */
export function atanh(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.atanh(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素向上取整: ceil(self) */
export function ceil(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.ceil(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素截断到 [min, max] 范围 */
export function clamp(
    self: Tensor,
    min?: number,
    max?: number,
    out?: Tensor
): Tensor {
    const result = internal.clamp(self._handle, min, max, out?._handle);
    return new Tensor(result);
}

/** 复数共轭: a+bi -> a-bi */
export function conj(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.conj(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素余弦: cos(self) */
export function cos(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.cos(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素双曲余弦: cosh(self) */
export function cosh(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.cosh(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素除法: self / other (标量版本) (Scalar variant) */
export function div(self: Tensor, other: number, roundingMode?: 'trunc' | 'floor'): Tensor;
/** 逐元素除法: self / other (Tensor variant) */
export function div(self: Tensor, other: Tensor, roundingMode?: 'trunc' | 'floor', out?: Tensor): Tensor;
export function div(
    self: Tensor,
    other: number | Tensor,
    roundingMode?: 'trunc' | 'floor',
    out?: Tensor
): Tensor | [Tensor, Tensor] {
    if (typeof other === 'number') {
        const result = internal.div_Scalar(self._handle, other, roundingMode);
        return new Tensor(result);
    } else {
        const result = internal.div_Tensor(self._handle, (other as Tensor)._handle, roundingMode, out?._handle);
        return new Tensor(result);
    }
}

/** 逐元素相等比较: self == other (标量版本) (Scalar variant) */
export function eq(self: Tensor, other: number): Tensor;
/** 逐元素相等比较: self == other (Tensor variant) */
export function eq(self: Tensor, other: Tensor): Tensor;
export function eq(
    self: Tensor,
    other: number | Tensor
): Tensor | [Tensor, Tensor] {
    if (typeof other === 'number') {
        const result = internal.eq_Scalar(self._handle, other);
        return new Tensor(result);
    } else {
        const result = internal.eq_Tensor(self._handle, (other as Tensor)._handle);
        return new Tensor(result);
    }
}

/** 逐元素误差函数: erf(self) */
export function erf(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.erf(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素互补误差函数: erfc(self) */
export function erfc(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.erfc(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素指数: e^self */
export function exp(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.exp(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素 2 的幂: 2^self */
export function exp2(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.exp2(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素 exp(self) - 1 */
export function expm1(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.expm1(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素向下取整: floor(self) */
export function floor(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.floor(self._handle, out?._handle);
    return new Tensor(result);
}

/** 向下取整除法: floor(self / other) (标量版本) (Scalar variant) */
export function floorDivide(self: Tensor, other: number, out?: Tensor): Tensor;
/** 向下取整除法: floor(self / other) (Tensor variant) */
export function floorDivide(self: Tensor, other: Tensor, out?: Tensor): Tensor;
export function floorDivide(
    self: Tensor,
    other: number | Tensor,
    out?: Tensor
): Tensor | [Tensor, Tensor] {
    if (typeof other === 'number') {
        const result = internal.floorDivide_Scalar(self._handle, other, out?._handle);
        return new Tensor(result);
    } else {
        const result = internal.floorDivide_Tensor(self._handle, (other as Tensor)._handle, out?._handle);
        return new Tensor(result);
    }
}

/** C++ 风格取模: fmod(self, other) (标量版本) (Scalar variant) */
export function fmod(self: Tensor, other: number, out?: Tensor): Tensor;
/** C++ 风格取模: fmod(self, other) (Tensor variant) */
export function fmod(self: Tensor, other: Tensor, out?: Tensor): Tensor;
export function fmod(
    self: Tensor,
    other: number | Tensor,
    out?: Tensor
): Tensor | [Tensor, Tensor] {
    if (typeof other === 'number') {
        const result = internal.fmod_Scalar(self._handle, other, out?._handle);
        return new Tensor(result);
    } else {
        const result = internal.fmod_Tensor(self._handle, (other as Tensor)._handle, out?._handle);
        return new Tensor(result);
    }
}

/** 逐元素取小数部分: frac(self) */
export function frac(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.frac(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素大于等于比较: self >= other (标量版本) (Scalar variant) */
export function ge(self: Tensor, other: number): Tensor;
/** 逐元素大于等于比较: self >= other (Tensor variant) */
export function ge(self: Tensor, other: Tensor): Tensor;
export function ge(
    self: Tensor,
    other: number | Tensor
): Tensor | [Tensor, Tensor] {
    if (typeof other === 'number') {
        const result = internal.ge_Scalar(self._handle, other);
        return new Tensor(result);
    } else {
        const result = internal.ge_Tensor(self._handle, (other as Tensor)._handle);
        return new Tensor(result);
    }
}

/** 逐元素大于比较: self > other (标量版本) (Scalar variant) */
export function gt(self: Tensor, other: number): Tensor;
/** 逐元素大于比较: self > other (Tensor variant) */
export function gt(self: Tensor, other: Tensor): Tensor;
export function gt(
    self: Tensor,
    other: number | Tensor
): Tensor | [Tensor, Tensor] {
    if (typeof other === 'number') {
        const result = internal.gt_Scalar(self._handle, other);
        return new Tensor(result);
    } else {
        const result = internal.gt_Tensor(self._handle, (other as Tensor)._handle);
        return new Tensor(result);
    }
}

/** 逐元素计算 0 阶修正贝塞尔函数: I₀(self) */
export function i0(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.i0(self._handle, out?._handle);
    return new Tensor(result);
}

/** 取虚部: a+bi -> b */
export function imag(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.imag(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素检查是否为有限数 */
export function isfinite(
    self: Tensor
): Tensor {
    const result = internal.isfinite(self._handle);
    return new Tensor(result);
}

/** 逐元素检查是否为无穷大 */
export function isinf(
    self: Tensor
): Tensor {
    const result = internal.isinf(self._handle);
    return new Tensor(result);
}

/** 逐元素检查是否为 NaN */
export function isnan(
    self: Tensor
): Tensor {
    const result = internal.isnan(self._handle);
    return new Tensor(result);
}

/** 逐元素小于等于比较: self <= other (标量版本) (Scalar variant) */
export function le(self: Tensor, other: number): Tensor;
/** 逐元素小于等于比较: self <= other (Tensor variant) */
export function le(self: Tensor, other: Tensor): Tensor;
export function le(
    self: Tensor,
    other: number | Tensor
): Tensor | [Tensor, Tensor] {
    if (typeof other === 'number') {
        const result = internal.le_Scalar(self._handle, other);
        return new Tensor(result);
    } else {
        const result = internal.le_Tensor(self._handle, (other as Tensor)._handle);
        return new Tensor(result);
    }
}

/** 逐元素自然对数: ln(self) */
export function log(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.log(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素以 10 为底对数: log10(self) */
export function log10(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.log10(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素 ln(1 + self) */
export function log1p(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.log1p(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素以 2 为底对数: log2(self) */
export function log2(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.log2(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素逻辑非: !self */
export function logicalNot(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.logicalNot(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素小于比较: self < other (标量版本) (Scalar variant) */
export function lt(self: Tensor, other: number): Tensor;
/** 逐元素小于比较: self < other (Tensor variant) */
export function lt(self: Tensor, other: Tensor): Tensor;
export function lt(
    self: Tensor,
    other: number | Tensor
): Tensor | [Tensor, Tensor] {
    if (typeof other === 'number') {
        const result = internal.lt_Scalar(self._handle, other);
        return new Tensor(result);
    } else {
        const result = internal.lt_Tensor(self._handle, (other as Tensor)._handle);
        return new Tensor(result);
    }
}

/** 逐元素最大值: max(self, other) */
export function maximum(
    self: Tensor,
    other: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.maximum(self._handle, other._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素最小值: min(self, other) */
export function minimum(
    self: Tensor,
    other: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.minimum(self._handle, other._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素乘法: self * other (标量版本) (Scalar variant) */
export function mul(self: Tensor, other: number): Tensor;
/** 逐元素乘法: self * other (Tensor variant) */
export function mul(self: Tensor, other: Tensor, out?: Tensor): Tensor;
export function mul(
    self: Tensor,
    other: number | Tensor,
    out?: Tensor
): Tensor | [Tensor, Tensor] {
    if (typeof other === 'number') {
        const result = internal.mul_Scalar(self._handle, other);
        return new Tensor(result);
    } else {
        const result = internal.mul_Tensor(self._handle, (other as Tensor)._handle, out?._handle);
        return new Tensor(result);
    }
}

/** 逐元素不等比较: self != other (标量版本) (Scalar variant) */
export function ne(self: Tensor, other: number): Tensor;
/** 逐元素不等比较: self != other (Tensor variant) */
export function ne(self: Tensor, other: Tensor): Tensor;
export function ne(
    self: Tensor,
    other: number | Tensor
): Tensor | [Tensor, Tensor] {
    if (typeof other === 'number') {
        const result = internal.ne_Scalar(self._handle, other);
        return new Tensor(result);
    } else {
        const result = internal.ne_Tensor(self._handle, (other as Tensor)._handle);
        return new Tensor(result);
    }
}

/** 逐元素取反: -self */
export function neg(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.neg(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素幂运算: self ^ exponent (标量版本) (Scalar variant) */
export function pow(self: Tensor, exponent: number, out?: Tensor): Tensor;
/** 逐元素幂运算: self ^ exponent (Tensor variant) */
export function pow(self: Tensor, exponent: Tensor, out?: Tensor): Tensor;
export function pow(
    self: Tensor,
    exponent: number | Tensor,
    out?: Tensor
): Tensor | [Tensor, Tensor] {
    if (typeof exponent === 'number') {
        const result = internal.pow_Scalar(self._handle, exponent, out?._handle);
        return new Tensor(result);
    } else {
        const result = internal.pow_Tensor(self._handle, (exponent as Tensor)._handle, out?._handle);
        return new Tensor(result);
    }
}

/** 取实部: a+bi -> a */
export function real(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.real(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素倒数: 1/self */
export function reciprocal(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.reciprocal(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素 ReLU: max(0, self) */
export function relu(
    self: Tensor
): Tensor {
    const result = internal.relu(self._handle);
    return new Tensor(result);
}

/** Python 风格取模: remainder(self, other) (标量版本) (Scalar variant) */
export function remainder(self: Tensor, other: number, out?: Tensor): Tensor;
/** Python 风格取模: remainder(self, other) (Tensor variant) */
export function remainder(self: Tensor, other: Tensor, out?: Tensor): Tensor;
export function remainder(
    self: Tensor,
    other: number | Tensor,
    out?: Tensor
): Tensor | [Tensor, Tensor] {
    if (typeof other === 'number') {
        const result = internal.remainder_Scalar(self._handle, other, out?._handle);
        return new Tensor(result);
    } else {
        const result = internal.remainder_Tensor(self._handle, (other as Tensor)._handle, out?._handle);
        return new Tensor(result);
    }
}

/** 逐元素四舍五入: round(self) */
export function round(
    self: Tensor,
    decimals?: number,
    out?: Tensor
): Tensor {
    const result = internal.round(self._handle, decimals, out?._handle);
    return new Tensor(result);
}

/** 逐元素平方根倒数: 1/sqrt(self) */
export function rsqrt(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.rsqrt(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素 Sigmoid: 1 / (1 + exp(-self)) */
export function sigmoid(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.sigmoid(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素符号: sign(self) */
export function sign(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.sign(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素正弦: sin(self) */
export function sin(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.sin(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素归一化 sinc: sin(πx)/(πx)，x=0 时返回 1 */
export function sinc(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.sinc(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素双曲正弦: sinh(self) */
export function sinh(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.sinh(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素平方根: sqrt(self) */
export function sqrt(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.sqrt(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素平方: self * self */
export function square(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.square(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素减法: self - alpha * other (标量版本) (Scalar variant) */
export function sub(self: Tensor, other: number, alpha?: number): Tensor;
/** 逐元素减法: self - alpha * other (Tensor variant) */
export function sub(self: Tensor, other: Tensor, alpha?: number, out?: Tensor): Tensor;
export function sub(
    self: Tensor,
    other: number | Tensor,
    alpha?: number,
    out?: Tensor
): Tensor | [Tensor, Tensor] {
    if (typeof other === 'number') {
        const result = internal.sub_Scalar(self._handle, other, alpha);
        return new Tensor(result);
    } else {
        const result = internal.sub_Tensor(self._handle, (other as Tensor)._handle, alpha, out?._handle);
        return new Tensor(result);
    }
}

/** 逐元素正切: tan(self) */
export function tan(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.tan(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素双曲正切: tanh(self) */
export function tanh(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.tanh(self._handle, out?._handle);
    return new Tensor(result);
}

/** 逐元素向零取整: trunc(self) */
export function trunc(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.trunc(self._handle, out?._handle);
    return new Tensor(result);
}

/** 根据条件选择: condition ? self : other */
export function where(
    condition: Tensor,
    self: Tensor,
    other: Tensor
): Tensor {
    const result = internal.where(condition._handle, self._handle, other._handle);
    return new Tensor(result);
}

export function dropout(
    self: Tensor,
    p?: number,
    training?: boolean
): Tensor {
    const result = internal.dropout(self._handle, p, training);
    return new Tensor(result);
}

export function elu(
    self: Tensor,
    alpha?: number
): Tensor {
    const result = internal.elu(self._handle, alpha);
    return new Tensor(result);
}

export function gelu(
    self: Tensor,
    approximate?: 'none' | 'tanh'
): Tensor {
    const result = internal.gelu(self._handle, approximate);
    return new Tensor(result);
}

export function hardtanh(
    self: Tensor,
    minVal?: number,
    maxVal?: number
): Tensor {
    const result = internal.hardtanh(self._handle, minVal, maxVal);
    return new Tensor(result);
}

export function leakyRelu(
    self: Tensor,
    negativeSlope?: number
): Tensor {
    const result = internal.leakyRelu(self._handle, negativeSlope);
    return new Tensor(result);
}

export function logSoftmax(
    self: Tensor,
    dim?: number,
    dtype?: DType
): Tensor {
    const result = internal.logSoftmax(self._handle, dim, dtype);
    return new Tensor(result);
}

/** LogSigmoid 激活: log(1 / (1 + exp(-self))) */
export function logsigmoid(
    self: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.logsigmoid(self._handle, out?._handle);
    return new Tensor(result);
}

/** SELU 激活 */
export function selu(
    self: Tensor,
    inplace?: boolean
): Tensor {
    const result = internal.selu(self._handle, inplace);
    return new Tensor(result);
}

export function silu(
    self: Tensor
): Tensor {
    const result = internal.silu(self._handle);
    return new Tensor(result);
}

export function softmax(
    self: Tensor,
    dim?: number,
    dtype?: DType
): Tensor {
    const result = internal.softmax(self._handle, dim, dtype);
    return new Tensor(result);
}

/** Softmin 激活: softmax(-self) */
export function softmin(
    self: Tensor,
    dim?: number,
    dtype?: DType
): Tensor {
    const result = internal.softmin(self._handle, dim, dtype);
    return new Tensor(result);
}

/** 沿维度逻辑与 */
export function all(
    self: Tensor,
    dim?: number,
    keepdim?: boolean
): Tensor {
    const result = internal.all(self._handle, dim, keepdim);
    return new Tensor(result);
}

/** 沿维度逻辑或 */
export function any(
    self: Tensor,
    dim?: number,
    keepdim?: boolean
): Tensor {
    const result = internal.any(self._handle, dim, keepdim);
    return new Tensor(result);
}

/** 沿维度最大值索引 */
export function argmax(
    self: Tensor,
    dim?: number,
    keepdim?: boolean
): Tensor {
    const result = internal.argmax(self._handle, dim, keepdim);
    return new Tensor(result);
}

/** 沿维度最小值索引 */
export function argmin(
    self: Tensor,
    dim?: number,
    keepdim?: boolean
): Tensor {
    const result = internal.argmin(self._handle, dim, keepdim);
    return new Tensor(result);
}

/** 数值稳定的 log(sum(exp(x)))，公式: max(x) + log(sum(exp(x - max(x)))) */
export function logsumexp(
    self: Tensor,
    dim: number | number[],
    keepdim?: boolean
): Tensor {
    const result = internal.logsumexp(self._handle, dim, keepdim);
    return new Tensor(result);
}

/** 沿维度最大值及索引 (dim variant) */
export function max(self: Tensor, dim: number, keepdim?: boolean): [Tensor, Tensor];
/** 全局最大值 (global variant) */
export function max(self: Tensor): Tensor;
export function max(
    self: Tensor,
    dim?: number,
    keepdim?: boolean
): Tensor | [Tensor, Tensor] {
    if (dim !== undefined) {
        const result = internal.max_dim(self._handle, dim, keepdim);
        return result.map(h => new Tensor(h)) as [Tensor, Tensor];
    } else {
        const result = internal.max_global(self._handle);
        return new Tensor(result);
    }
}

/** 沿维度求均值 */
export function mean(
    self: Tensor,
    dim?: number | number[],
    keepdim?: boolean,
    dtype?: DType
): Tensor {
    const result = internal.mean(self._handle, dim, keepdim, dtype);
    return new Tensor(result);
}

/** 沿维度最小值及索引 (dim variant) */
export function min(self: Tensor, dim: number, keepdim?: boolean): [Tensor, Tensor];
/** 全局最小值 (global variant) */
export function min(self: Tensor): Tensor;
export function min(
    self: Tensor,
    dim?: number,
    keepdim?: boolean
): Tensor | [Tensor, Tensor] {
    if (dim !== undefined) {
        const result = internal.min_dim(self._handle, dim, keepdim);
        return result.map(h => new Tensor(h)) as [Tensor, Tensor];
    } else {
        const result = internal.min_global(self._handle);
        return new Tensor(result);
    }
}

/** 沿维度求均值(忽略NaN) */
export function nanmean(
    self: Tensor,
    dim?: number | number[],
    keepdim?: boolean,
    dtype?: DType
): Tensor {
    const result = internal.nanmean(self._handle, dim, keepdim, dtype);
    return new Tensor(result);
}

/** 沿维度求和(忽略NaN) */
export function nansum(
    self: Tensor,
    dim?: number | number[],
    keepdim?: boolean,
    dtype?: DType
): Tensor {
    const result = internal.nansum(self._handle, dim, keepdim, dtype);
    return new Tensor(result);
}

/** 沿维度范数 */
export function norm(
    self: Tensor,
    p?: number,
    dim?: number | number[],
    keepdim?: boolean
): Tensor {
    const result = internal.norm(self._handle, p, dim, keepdim);
    return new Tensor(result);
}

/** 沿维度求积 */
export function prod(
    self: Tensor,
    dim?: number,
    keepdim?: boolean,
    dtype?: DType
): Tensor {
    const result = internal.prod(self._handle, dim, keepdim, dtype);
    return new Tensor(result);
}

/** 沿维度标准差 */
export function std(
    self: Tensor,
    dim?: number | number[],
    correction?: number,
    keepdim?: boolean
): Tensor {
    const result = internal.std(self._handle, dim, correction, keepdim);
    return new Tensor(result);
}

/** 沿维度求和 */
export function sum(
    self: Tensor,
    dim?: number | number[],
    keepdim?: boolean,
    dtype?: DType
): Tensor {
    const result = internal.sum(self._handle, dim, keepdim, dtype);
    return new Tensor(result);
}

/** 沿维度方差 */
export function variance(
    self: Tensor,
    dim?: number | number[],
    correction?: number,
    keepdim?: boolean
): Tensor {
    const result = internal.variance(self._handle, dim, correction, keepdim);
    return new Tensor(result);
}

/** out = beta * self + alpha * (mat1 @ mat2) */
export function addmm(
    self: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    beta?: number,
    alpha?: number,
    out?: Tensor
): Tensor {
    const result = internal.addmm(self._handle, mat1._handle, mat2._handle, beta, alpha, out?._handle);
    return new Tensor(result);
}

/** out = beta * self + alpha * (mat @ vec) */
export function addmv(
    self: Tensor,
    mat: Tensor,
    vec: Tensor,
    beta?: number,
    alpha?: number,
    out?: Tensor
): Tensor {
    const result = internal.addmv(self._handle, mat._handle, vec._handle, beta, alpha, out?._handle);
    return new Tensor(result);
}

/** out = beta * self + alpha * (batch1 @ batch2) */
export function baddbmm(
    self: Tensor,
    batch1: Tensor,
    batch2: Tensor,
    beta?: number,
    alpha?: number,
    out?: Tensor
): Tensor {
    const result = internal.baddbmm(self._handle, batch1._handle, batch2._handle, beta, alpha, out?._handle);
    return new Tensor(result);
}

/** 批量矩阵乘法 */
export function bmm(
    self: Tensor,
    mat2: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.bmm(self._handle, mat2._handle, out?._handle);
    return new Tensor(result);
}

/** 向量点积 */
export function dot(
    self: Tensor,
    other: Tensor
): Tensor {
    const result = internal.dot(self._handle, other._handle);
    return new Tensor(result);
}

/** 线性变换: y = input @ weight.T + bias */
export function linear(
    input: Tensor,
    weight: Tensor,
    bias?: Tensor
): Tensor {
    const result = internal.linear(input._handle, weight._handle, bias?._handle);
    return new Tensor(result);
}

/** 矩阵乘法 (支持批量) */
export function matmul(
    self: Tensor,
    other: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.matmul(self._handle, other._handle, out?._handle);
    return new Tensor(result);
}

/** 2D 矩阵乘法 */
export function mm(
    self: Tensor,
    mat2: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.mm(self._handle, mat2._handle, out?._handle);
    return new Tensor(result);
}

/** 矩阵-向量乘法 */
export function mv(
    self: Tensor,
    vec: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.mv(self._handle, vec._handle, out?._handle);
    return new Tensor(result);
}

/** 向量外积 */
export function outer(
    self: Tensor,
    vec2: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.outer(self._handle, vec2._handle, out?._handle);
    return new Tensor(result);
}

/** 对角线构造 (1D->2D) 或提取 (2D->1D) */
export function diag(
    self: Tensor,
    diagonal?: number
): Tensor {
    const result = internal.diag(self._handle, diagonal);
    return new Tensor(result);
}

/** 获取对角线视图 (Partial View) */
export function diagonal(
    self: Tensor,
    offset?: number,
    dim1?: number,
    dim2?: number
): Tensor {
    const result = internal.diagonal(self._handle, offset, dim1, dim2);
    return new Tensor(result);
}

/** 矩阵迹: 对角线元素之和 */
export function trace(
    self: Tensor
): Tensor {
    const result = internal.trace(self._handle);
    return new Tensor(result);
}

/** 下三角矩阵: 保留 row >= col + diagonal 的元素，其余置零 */
export function tril(
    self: Tensor,
    diagonal?: number
): Tensor {
    const result = internal.tril(self._handle, diagonal);
    return new Tensor(result);
}

/** 上三角矩阵: 保留 row <= col + diagonal 的元素，其余置零 */
export function triu(
    self: Tensor,
    diagonal?: number
): Tensor {
    const result = internal.triu(self._handle, diagonal);
    return new Tensor(result);
}

/** Batch Normalization */
export function batchNorm(
    self: Tensor,
    runningMean?: Tensor,
    runningVar?: Tensor,
    weight?: Tensor,
    bias?: Tensor,
    training?: boolean,
    momentum?: number,
    eps?: number
): Tensor {
    const result = internal.batchNorm(self._handle, runningMean?._handle, runningVar?._handle, weight?._handle, bias?._handle, training, momentum, eps);
    return new Tensor(result);
}

/** Group Normalization */
export function groupNorm(
    self: Tensor,
    numGroups: number,
    weight?: Tensor,
    bias?: Tensor,
    eps?: number
): Tensor {
    const result = internal.groupNorm(self._handle, numGroups, weight?._handle, bias?._handle, eps);
    return new Tensor(result);
}

/** Layer Normalization */
export function layerNorm(
    self: Tensor,
    normalizedShape: number[],
    weight?: Tensor,
    bias?: Tensor,
    eps?: number
): Tensor {
    const result = internal.layerNorm(self._handle, normalizedShape, weight?._handle, bias?._handle, eps);
    return new Tensor(result);
}

/** F.normalize: self / self.norm() */
export function normalize(
    self: Tensor,
    p?: number,
    dim?: number,
    eps?: number,
    out?: Tensor
): Tensor {
    const result = internal.normalize(self._handle, p, dim, eps, out?._handle);
    return new Tensor(result);
}

/** RMS Normalization */
export function rmsNorm(
    self: Tensor,
    normalizedShape: number[],
    weight?: Tensor,
    eps?: number
): Tensor {
    const result = internal.rmsNorm(self._handle, normalizedShape, weight?._handle, eps);
    return new Tensor(result);
}

/** 创建具有指定 size 和 stride 的视图（STFT 分帧等场景的核心操作） */
export function asStrided(
    self: Tensor,
    size: number[],
    stride: number[],
    storageOffset?: number
): Tensor {
    const result = internal.asStrided(self._handle, size, stride, storageOffset);
    return new Tensor(result);
}

/** 沿指定维度拼接张量序列 */
export function cat(
    tensors: Tensor[],
    dim?: number,
    out?: Tensor
): Tensor {
    const result = internal.cat(tensors.map(t => t._handle), dim, out?._handle);
    return new Tensor(result);
}

/** 计算 N 阶前向差分: out[i] = input[i+1] - input[i] */
export function diff(
    self: Tensor,
    n?: number,
    dim?: number,
    prepend?: Tensor,
    append?: Tensor
): Tensor {
    const result = internal.diff(self._handle, n, dim, prepend?._handle, append?._handle);
    return new Tensor(result);
}

export function expand(
    self: Tensor,
    size: number[]
): Tensor {
    const result = internal.expand(self._handle, size);
    return new Tensor(result);
}

export function flatten(
    self: Tensor,
    startDim?: number,
    endDim?: number
): Tensor {
    const result = internal.flatten(self._handle, startDim, endDim);
    return new Tensor(result);
}

/** 沿给定维度翻转张量元素顺序 */
export function flip(
    self: Tensor,
    dims: number | number[]
): Tensor {
    const result = internal.flip(self._handle, dims);
    return new Tensor(result);
}

/** 左右翻转 (沿 dim=1 翻转)，要求至少 2D */
export function fliplr(
    self: Tensor
): Tensor {
    const result = internal.fliplr(self._handle);
    return new Tensor(result);
}

/** 上下翻转 (沿 dim=0 翻转)，要求至少 1D */
export function flipud(
    self: Tensor
): Tensor {
    const result = internal.flipud(self._handle);
    return new Tensor(result);
}

export function permute(
    self: Tensor,
    dims: number | number[]
): Tensor {
    const result = internal.permute(self._handle, dims);
    return new Tensor(result);
}

/** 沿维度重复每个元素指定次数 */
export function repeatInterleave(
    self: Tensor,
    repeats: number | Tensor,
    dim?: number,
    outputSize?: number
): Tensor {
    const result = internal.repeatInterleave(self._handle, (repeats instanceof Tensor ? repeats._handle : repeats), dim, outputSize);
    return new Tensor(result);
}

export function reshape(
    self: Tensor,
    shape: number[]
): Tensor {
    const result = internal.reshape(self._handle, shape);
    return new Tensor(result);
}

/** 沿指定维度选择单个索引，返回降维的视图 */
export function select(
    self: Tensor,
    dim: number,
    index: number
): Tensor {
    const result = internal.select(self._handle, dim, index);
    return new Tensor(result);
}

/** 张量切片: 使用 Python 风格切片语法或整数索引返回视图 */
export function slice(
    self: Tensor,
    slices: string | number
): Tensor {
    const result = internal.slice(self._handle, slices);
    return new Tensor(result);
}

export function squeeze(
    self: Tensor,
    dim?: number
): Tensor {
    const result = internal.squeeze(self._handle, dim);
    return new Tensor(result);
}

/** 沿新维度堆叠张量序列 */
export function stack(
    tensors: Tensor[],
    dim?: number,
    out?: Tensor
): Tensor {
    const result = internal.stack(tensors.map(t => t._handle), dim, out?._handle);
    return new Tensor(result);
}

export function transpose(
    self: Tensor,
    dim0: number,
    dim1: number
): Tensor {
    const result = internal.transpose(self._handle, dim0, dim1);
    return new Tensor(result);
}

export function unsqueeze(
    self: Tensor,
    dim: number
): Tensor {
    const result = internal.unsqueeze(self._handle, dim);
    return new Tensor(result);
}

export function view(
    self: Tensor,
    shape: number[]
): Tensor {
    const result = internal.view(self._handle, shape);
    return new Tensor(result);
}

export function arange(
    start: number,
    end?: number,
    step?: number,
    dtype?: DType,
    device?: string
): Tensor {
    const result = internal.arange(start, end, step, dtype, device);
    return new Tensor(result);
}

/** 生成 Bartlett 窗函数 */
export function bartlettWindow(
    windowLength: number,
    periodic?: boolean,
    dtype?: DType,
    device?: string
): Tensor {
    const result = internal.bartlettWindow(windowLength, periodic, dtype, device);
    return new Tensor(result);
}

/** 生成 Blackman 窗函数 */
export function blackmanWindow(
    windowLength: number,
    periodic?: boolean,
    dtype?: DType,
    device?: string
): Tensor {
    const result = internal.blackmanWindow(windowLength, periodic, dtype, device);
    return new Tensor(result);
}

export function empty(
    size: number[],
    dtype?: DType,
    device?: string
): Tensor {
    const result = internal.empty(size, dtype, device);
    return new Tensor(result);
}

export function emptyLike(
    self: Tensor,
    dtype?: DType,
    device?: string
): Tensor {
    const result = internal.emptyLike(self._handle, dtype, device);
    return new Tensor(result);
}

export function eye(
    n: number,
    m?: number,
    dtype?: DType,
    device?: string
): Tensor {
    const result = internal.eye(n, m, dtype, device);
    return new Tensor(result);
}

export function full(
    size: number[],
    fillValue: number,
    dtype?: DType,
    device?: string
): Tensor {
    const result = internal.full(size, fillValue, dtype, device);
    return new Tensor(result);
}

/** 生成 Hamming 窗函数 */
export function hammingWindow(
    windowLength: number,
    periodic?: boolean,
    alpha?: number,
    beta?: number,
    dtype?: DType,
    device?: string
): Tensor {
    const result = internal.hammingWindow(windowLength, periodic, alpha, beta, dtype, device);
    return new Tensor(result);
}

/** 生成 Hann 窗函数 */
export function hannWindow(
    windowLength: number,
    periodic?: boolean,
    dtype?: DType,
    device?: string
): Tensor {
    const result = internal.hannWindow(windowLength, periodic, dtype, device);
    return new Tensor(result);
}

/** 生成 Kaiser 窗函数 (需要 Bessel I₀ 函数) */
export function kaiserWindow(
    windowLength: number,
    periodic?: boolean,
    beta?: number,
    dtype?: DType,
    device?: string
): Tensor {
    const result = internal.kaiserWindow(windowLength, periodic, beta, dtype, device);
    return new Tensor(result);
}

export function linspace(
    start: number,
    end: number,
    steps: number,
    dtype?: DType,
    device?: string
): Tensor {
    const result = internal.linspace(start, end, steps, dtype, device);
    return new Tensor(result);
}

/** 从多项式分布中采样索引 */
export function multinomial(
    input: Tensor,
    numSamples: number,
    replacement?: boolean
): Tensor {
    const result = internal.multinomial(input._handle, numSamples, replacement);
    return new Tensor(result);
}

export function ones(
    size: number[],
    dtype?: DType,
    device?: string
): Tensor {
    const result = internal.ones(size, dtype, device);
    return new Tensor(result);
}

export function onesLike(
    self: Tensor,
    dtype?: DType,
    device?: string
): Tensor {
    const result = internal.onesLike(self._handle, dtype, device);
    return new Tensor(result);
}

/** N 维张量填充 (STFT center padding 等场景) */
export function pad(
    input: Tensor,
    pad: number[],
    mode?: 'constant' | 'reflect' | 'replicate' | 'circular',
    value?: number
): Tensor {
    const result = internal.pad(input._handle, pad, mode, value);
    return new Tensor(result);
}

export function rand(
    size: number[],
    dtype?: DType,
    device?: string
): Tensor {
    const result = internal.rand(size, dtype, device);
    return new Tensor(result);
}

export function randint(
    low: number,
    high: number,
    size: number[],
    dtype?: DType,
    device?: string
): Tensor {
    const result = internal.randint(low, high, size, dtype, device);
    return new Tensor(result);
}

export function randn(
    size: number[],
    dtype?: DType,
    device?: string
): Tensor {
    const result = internal.randn(size, dtype, device);
    return new Tensor(result);
}

export function zeros(
    size: number[],
    dtype?: DType,
    device?: string
): Tensor {
    const result = internal.zeros(size, dtype, device);
    return new Tensor(result);
}

export function zerosLike(
    self: Tensor,
    dtype?: DType,
    device?: string
): Tensor {
    const result = internal.zerosLike(self._handle, dtype, device);
    return new Tensor(result);
}

/** 类型转换: 将 tensor 转换为指定的 dtype */
export function cast(
    self: Tensor,
    dtype: DType
): Tensor {
    const result = internal.cast(self._handle, dtype);
    return new Tensor(result);
}

export function clone(
    self: Tensor
): Tensor {
    const result = internal.clone(self._handle);
    return new Tensor(result);
}

/** 确保 tensor 按指定格式连续存储。如果已是目标格式则返回原 tensor。 */
export function contiguous(
    self: Tensor,
    memoryFormat?: unknown
): Tensor {
    const result = internal.contiguous(self._handle, memoryFormat);
    return new Tensor(result);
}

export function to(
    self: Tensor,
    dtype?: DType,
    device?: string,
    copy?: boolean
): Tensor {
    const result = internal.to(self._handle, dtype, device, copy);
    return new Tensor(result);
}

export function cummax(
    self: Tensor,
    dim: number
): [Tensor, Tensor] {
    const result = internal.cummax(self._handle, dim);
    return result.map(h => new Tensor(h)) as [Tensor, Tensor];
}

export function cummin(
    self: Tensor,
    dim: number
): [Tensor, Tensor] {
    const result = internal.cummin(self._handle, dim);
    return result.map(h => new Tensor(h)) as [Tensor, Tensor];
}

export function cumprod(
    self: Tensor,
    dim: number,
    dtype?: DType
): Tensor {
    const result = internal.cumprod(self._handle, dim, dtype);
    return new Tensor(result);
}

export function cumsum(
    self: Tensor,
    dim: number,
    dtype?: DType
): Tensor {
    const result = internal.cumsum(self._handle, dim, dtype);
    return new Tensor(result);
}

export function argsort(
    self: Tensor,
    dim?: number,
    descending?: boolean,
    stable?: boolean
): Tensor {
    const result = internal.argsort(self._handle, dim, descending, stable);
    return new Tensor(result);
}

export function sort(
    self: Tensor,
    dim?: number,
    descending?: boolean,
    stable?: boolean
): [Tensor, Tensor] {
    const result = internal.sort(self._handle, dim, descending, stable);
    return result.map(h => new Tensor(h)) as [Tensor, Tensor];
}

export function topk(
    self: Tensor,
    k: number,
    dim?: number,
    largest?: boolean,
    sorted?: boolean
): [Tensor, Tensor] {
    const result = internal.topk(self._handle, k, dim, largest, sorted);
    return result.map(h => new Tensor(h)) as [Tensor, Tensor];
}

/** 自适应 2D 平均池化（自动计算 kernel/stride） */
export function adaptiveAvgPool2d(
    input: Tensor,
    outputSize: number | number[]
): Tensor {
    const result = internal.adaptiveAvgPool2d(input._handle, outputSize);
    return new Tensor(result);
}

/** 自适应 2D 最大池化 */
export function adaptiveMaxPool2d(
    input: Tensor,
    outputSize: number | number[],
    returnIndices?: boolean
): Tensor {
    const result = internal.adaptiveMaxPool2d(input._handle, outputSize, returnIndices);
    return new Tensor(result);
}

/** 1D 平均池化 */
export function avgPool1d(
    input: Tensor,
    kernelSize: number,
    stride?: number,
    padding?: number,
    ceilMode?: boolean,
    countIncludePad?: boolean
): Tensor {
    const result = internal.avgPool1d(input._handle, kernelSize, stride, padding, ceilMode, countIncludePad);
    return new Tensor(result);
}

/** 2D 平均池化 */
export function avgPool2d(
    input: Tensor,
    kernelSize: number | number[],
    stride?: number | number[],
    padding?: number | number[],
    ceilMode?: boolean,
    countIncludePad?: boolean,
    divisorOverride?: number
): Tensor {
    const result = internal.avgPool2d(input._handle, kernelSize, stride, padding, ceilMode, countIncludePad, divisorOverride);
    return new Tensor(result);
}

/** 3D 平均池化 */
export function avgPool3d(
    input: Tensor,
    kernelSize: number | number[],
    stride?: number | number[],
    padding?: number | number[],
    ceilMode?: boolean,
    countIncludePad?: boolean,
    divisorOverride?: number
): Tensor {
    const result = internal.avgPool3d(input._handle, kernelSize, stride, padding, ceilMode, countIncludePad, divisorOverride);
    return new Tensor(result);
}

/** 1D 卷积操作 */
export function conv1d(
    input: Tensor,
    weight: Tensor,
    bias?: Tensor,
    stride?: number,
    padding?: number | 'same' | 'valid',
    dilation?: number,
    groups?: number
): Tensor {
    const result = internal.conv1d(input._handle, weight._handle, bias?._handle, stride, padding, dilation, groups);
    return new Tensor(result);
}

/** 2D 卷积操作 (PyTorch F.conv2d 兼容) */
export function conv2d(
    input: Tensor,
    weight: Tensor,
    bias?: Tensor,
    stride?: number | number[],
    padding?: number | number[] | 'same' | 'valid',
    dilation?: number | number[],
    groups?: number
): Tensor {
    const result = internal.conv2d(input._handle, weight._handle, bias?._handle, stride, padding, dilation, groups);
    return new Tensor(result);
}

/** 3D 卷积操作 */
export function conv3d(
    input: Tensor,
    weight: Tensor,
    bias?: Tensor,
    stride?: number | number[],
    padding?: number | number[] | 'same' | 'valid',
    dilation?: number | number[],
    groups?: number
): Tensor {
    const result = internal.conv3d(input._handle, weight._handle, bias?._handle, stride, padding, dilation, groups);
    return new Tensor(result);
}

/** 2D 转置卷积 (反卷积) */
export function convTranspose2d(
    input: Tensor,
    weight: Tensor,
    bias?: Tensor,
    stride?: number | number[],
    padding?: number | number[],
    outputPadding?: number | number[],
    groups?: number,
    dilation?: number | number[]
): Tensor {
    const result = internal.convTranspose2d(input._handle, weight._handle, bias?._handle, stride, padding, outputPadding, groups, dilation);
    return new Tensor(result);
}

/** 1D 最大池化 */
export function maxPool1d(
    input: Tensor,
    kernelSize: number,
    stride?: number,
    padding?: number,
    dilation?: number,
    ceilMode?: boolean,
    returnIndices?: boolean
): Tensor {
    const result = internal.maxPool1d(input._handle, kernelSize, stride, padding, dilation, ceilMode, returnIndices);
    return new Tensor(result);
}

/** 2D 最大池化 */
export function maxPool2d(
    input: Tensor,
    kernelSize: number | number[],
    stride?: number | number[],
    padding?: number | number[],
    dilation?: number | number[],
    ceilMode?: boolean,
    returnIndices?: boolean
): Tensor;
export function maxPool2d(
    input: Tensor,
    kernelSize: number | number[],
    stride?: number | number[],
    padding?: number | number[],
    dilation?: number | number[],
    ceilMode?: boolean,
    returnIndices?: boolean
): [Tensor, Tensor];
export function maxPool2d(
    input: Tensor,
    kernelSize: number | number[],
    stride?: number | number[],
    padding?: number | number[],
    dilation?: number | number[],
    ceilMode?: boolean,
    returnIndices?: boolean
): Tensor | [Tensor, Tensor] {
    const result = internal.maxPool2d(input._handle, kernelSize, stride, padding, dilation, ceilMode, returnIndices);
    // Handle conditional return based on returnIndices
    if (Array.isArray(result)) {
        return result.map(h => new Tensor(h)) as [Tensor, Tensor];
    }
    return new Tensor(result);
}

/** 3D 最大池化 */
export function maxPool3d(
    input: Tensor,
    kernelSize: number | number[],
    stride?: number | number[],
    padding?: number | number[],
    dilation?: number | number[],
    ceilMode?: boolean,
    returnIndices?: boolean
): Tensor {
    const result = internal.maxPool3d(input._handle, kernelSize, stride, padding, dilation, ceilMode, returnIndices);
    return new Tensor(result);
}

/** 嵌入查找: output[...] = weight[input[...], :] */
export function embedding(
    input: Tensor,
    weight: Tensor,
    paddingIdx?: number,
    maxNorm?: number,
    normType?: number,
    scaleGradByFreq?: boolean,
    sparse?: boolean
): Tensor {
    const result = internal.embedding(input._handle, weight._handle, paddingIdx, maxNorm, normType, scaleGradByFreq, sparse);
    return new Tensor(result);
}

/** 沿维度选择索引: out[...] = self.select(dim, index[...]) */
export function indexSelect(
    self: Tensor,
    dim: number,
    index: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.indexSelect(self._handle, dim, index._handle, out?._handle);
    return new Tensor(result);
}

/** 散射操作: out[index[...]][...] = src[...] */
export function scatter(
    self: Tensor,
    dim: number,
    index: Tensor,
    src: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.scatter(self._handle, dim, index._handle, src._handle, out?._handle);
    return new Tensor(result);
}

/** 散射加法: out[index[...]][...] += src[...] */
export function scatterAdd(
    self: Tensor,
    dim: number,
    index: Tensor,
    src: Tensor,
    out?: Tensor
): Tensor {
    const result = internal.scatterAdd(self._handle, dim, index._handle, src._handle, out?._handle);
    return new Tensor(result);
}

/** 通用散射归约 */
export function scatterReduce(
    self: Tensor,
    dim: number,
    index: Tensor,
    src: Tensor,
    reduce: 'sum' | 'prod' | 'mean' | 'amax' | 'amin',
    includeSelf?: boolean,
    out?: Tensor
): Tensor {
    const result = internal.scatterReduce(self._handle, dim, index._handle, src._handle, reduce, includeSelf, out?._handle);
    return new Tensor(result);
}

/**
 * 原地拷贝: 将 src 拷贝到 self (in-place)
 * 
 * @param self 目标张量 (会被原地修改)
 * @param src 源张量
 * @returns self (返回修改后的 self 以支持链式调用)
 */
export function copy_(
    self: Tensor,
    src: Tensor
): Tensor {
    const result = internal.copy_(self._handle, src._handle);
    return new Tensor(result);
}

/** Scaled Dot-Product Attention: softmax(Q @ K^T / scale + mask) @ V */
export function scaledDotProductAttention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attnMask?: Tensor,
    dropoutP?: number,
    isCausal?: boolean,
    scale?: number
): Tensor {
    const result = internal.scaledDotProductAttention(query._handle, key._handle, value._handle, attnMask?._handle, dropoutP, isCausal, scale);
    return new Tensor(result);
}

/** Computes the one dimensional discrete Fourier transform of input. (internal: fftImpl to avoid namespace conflict) */
export function fftImpl(
    input: Tensor,
    n?: number,
    dim?: number,
    norm?: 'forward' | 'backward' | 'ortho'
): Tensor {
    const result = internal.fft(input._handle, n, dim, norm);
    return new Tensor(result);
}

/** Computes the 2-dimensional discrete Fourier transform of input. */
export function fft2(
    input: Tensor,
    s?: number[],
    dim?: number | number[],
    norm?: 'forward' | 'backward' | 'ortho'
): Tensor {
    const result = internal.fft2(input._handle, s, dim, norm);
    return new Tensor(result);
}

/** Returns the DFT sample frequencies. */
export function fftfreq(
    n: number,
    d?: number
): Tensor {
    const result = internal.fftfreq(n, d);
    return new Tensor(result);
}

/** Computes the N-dimensional discrete Fourier transform of input. */
export function fftn(
    input: Tensor,
    s?: number[],
    dim?: number | number[],
    norm?: 'forward' | 'backward' | 'ortho'
): Tensor {
    const result = internal.fftn(input._handle, s, dim, norm);
    return new Tensor(result);
}

/** Shift zero-frequency component to center of spectrum. */
export function fftshift(
    input: Tensor,
    dim?: number | number[]
): Tensor {
    const result = internal.fftshift(input._handle, dim);
    return new Tensor(result);
}

/** Computes the 1D discrete Fourier transform of a Hermitian symmetric input signal. */
export function hfft(
    input: Tensor,
    n?: number,
    dim?: number,
    norm?: 'forward' | 'backward' | 'ortho'
): Tensor {
    const result = internal.hfft(input._handle, n, dim, norm);
    return new Tensor(result);
}

/** Computes the one dimensional inverse discrete Fourier transform of input. */
export function ifft(
    input: Tensor,
    n?: number,
    dim?: number,
    norm?: 'forward' | 'backward' | 'ortho'
): Tensor {
    const result = internal.ifft(input._handle, n, dim, norm);
    return new Tensor(result);
}

/** Computes the 2-dimensional inverse discrete Fourier transform of input. */
export function ifft2(
    input: Tensor,
    s?: number[],
    dim?: number | number[],
    norm?: 'forward' | 'backward' | 'ortho'
): Tensor {
    const result = internal.ifft2(input._handle, s, dim, norm);
    return new Tensor(result);
}

/** Computes the N-dimensional inverse discrete Fourier transform of input. */
export function ifftn(
    input: Tensor,
    s?: number[],
    dim?: number | number[],
    norm?: 'forward' | 'backward' | 'ortho'
): Tensor {
    const result = internal.ifftn(input._handle, s, dim, norm);
    return new Tensor(result);
}

/** Inverse of fftshift. */
export function ifftshift(
    input: Tensor,
    dim?: number | number[]
): Tensor {
    const result = internal.ifftshift(input._handle, dim);
    return new Tensor(result);
}

/** Computes the inverse of hfft. */
export function ihfft(
    input: Tensor,
    n?: number,
    dim?: number,
    norm?: 'forward' | 'backward' | 'ortho'
): Tensor {
    const result = internal.ihfft(input._handle, n, dim, norm);
    return new Tensor(result);
}

/** Computes the inverse FFT of rfft. Output is real-valued. */
export function irfft(
    input: Tensor,
    n?: number,
    dim?: number,
    norm?: 'forward' | 'backward' | 'ortho'
): Tensor {
    const result = internal.irfft(input._handle, n, dim, norm);
    return new Tensor(result);
}

/** Computes the inverse 2D FFT of rfft2 output. Output is real-valued. */
export function irfft2(
    input: Tensor,
    s?: number[],
    dim?: number | number[],
    norm?: 'forward' | 'backward' | 'ortho'
): Tensor {
    const result = internal.irfft2(input._handle, s, dim, norm);
    return new Tensor(result);
}

/** Computes the N-dimensional inverse FFT of real input. */
export function irfftn(
    input: Tensor,
    s?: number[],
    dim?: number | number[],
    norm?: 'forward' | 'backward' | 'ortho'
): Tensor {
    const result = internal.irfftn(input._handle, s, dim, norm);
    return new Tensor(result);
}

/** Computes the inverse Short-Time Fourier Transform to reconstruct the signal. */
export function istft(
    input: Tensor,
    n_fft: number,
    hop_length?: number,
    win_length?: number,
    window?: Tensor,
    center?: boolean,
    normalized?: boolean,
    onesided?: boolean,
    length?: number,
    return_complex?: boolean
): Tensor {
    const result = internal.istft(input._handle, n_fft, hop_length, win_length, window?._handle, center, normalized, onesided, length, return_complex);
    return new Tensor(result);
}

/** Computes the one dimensional FFT of real-valued input. */
export function rfft(
    input: Tensor,
    n?: number,
    dim?: number,
    norm?: 'forward' | 'backward' | 'ortho'
): Tensor {
    const result = internal.rfft(input._handle, n, dim, norm);
    return new Tensor(result);
}

/** Computes the 2-dimensional FFT of real-valued input. */
export function rfft2(
    input: Tensor,
    s?: number[],
    dim?: number | number[],
    norm?: 'forward' | 'backward' | 'ortho'
): Tensor {
    const result = internal.rfft2(input._handle, s, dim, norm);
    return new Tensor(result);
}

/** Returns the sample frequencies for rfft. */
export function rfftfreq(
    n: number,
    d?: number
): Tensor {
    const result = internal.rfftfreq(n, d);
    return new Tensor(result);
}

/** Computes the N-dimensional FFT of real-valued input. */
export function rfftn(
    input: Tensor,
    s?: number[],
    dim?: number | number[],
    norm?: 'forward' | 'backward' | 'ortho'
): Tensor {
    const result = internal.rfftn(input._handle, s, dim, norm);
    return new Tensor(result);
}

/** Computes the Short-Time Fourier Transform of the input signal. */
export function stft(
    input: Tensor,
    n_fft: number,
    hop_length?: number,
    win_length?: number,
    window?: Tensor,
    center?: boolean,
    pad_mode?: 'constant' | 'reflect' | 'replicate' | 'circular',
    normalized?: boolean,
    onesided?: boolean,
    return_complex?: boolean
): Tensor {
    const result = internal.stft(input._handle, n_fft, hop_length, win_length, window?._handle, center, pad_mode, normalized, onesided, return_complex);
    return new Tensor(result);
}

// ============================================================================
// nn.functional namespace
// ============================================================================

/**
 * Functional interface for neural network operations
 * Provides PyTorch-compatible nn.functional.* API
 * @see https://pytorch.org/docs/stable/nn.functional.html
 */
export const functional = {
    adaptiveAvgPool2d,
    adaptiveMaxPool2d,
    avgPool1d,
    avgPool2d,
    avgPool3d,
    batchNorm,
    conv1d,
    conv2d,
    conv3d,
    convTranspose2d,
    dropout,
    elu,
    embedding,
    gelu,
    groupNorm,
    hardtanh,
    layerNorm,
    leakyRelu,
    linear,
    logSoftmax,
    logsigmoid,
    maxPool1d,
    maxPool2d,
    maxPool3d,
    normalize,
    relu,
    rmsNorm,
    scaledDotProductAttention,
    selu,
    sigmoid,
    silu,
    softmax,
    softmin,
    tanh,
} as const;

// ============================================================================
// fft namespace (torch.fft.*)
// ============================================================================

/**
 * FFT operations namespace
 * Provides PyTorch-compatible torch.fft.* API
 * @see https://pytorch.org/docs/stable/fft.html
 */
export const fft = {
    fft: fftImpl,
    fft2,
    fftfreq,
    fftn,
    fftshift,
    hfft,
    ifft,
    ifft2,
    ifftn,
    ifftshift,
    ihfft,
    irfft,
    irfft2,
    irfftn,
    rfft,
    rfft2,
    rfftfreq,
    rfftn,
} as const;
