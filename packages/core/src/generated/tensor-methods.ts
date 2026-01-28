/**
 * v5 Generated Tensor Methods
 * DO NOT EDIT - Generated from OpRegistry
 */

import { DType } from '@kandle/types';
import { Tensor } from '../tensor';
import * as ops from './ops';

// ============================================================================
// TypeScript Module Augmentation
// ============================================================================

declare module '../tensor' {
    interface Tensor<T extends DType = DType> {
        /** 逐元素绝对值: |self| */
        abs(out?: Tensor): Tensor;
        /** 逐元素反余弦: arccos(self) */
        acos(out?: Tensor): Tensor;
        /** 逐元素反双曲余弦: arccosh(self) */
        acosh(out?: Tensor): Tensor;
        /** 逐元素加法: self + alpha * other (标量版本) */
        add(other: number | Tensor, alpha?: number, out?: Tensor): Tensor;
        /** 逐元素计算复数相位角 (弧度)，实数返回 0 或 π */
        angle(out?: Tensor): Tensor;
        /** 逐元素反正弦: arcsin(self) */
        asin(out?: Tensor): Tensor;
        /** 逐元素反双曲正弦: arcsinh(self) */
        asinh(out?: Tensor): Tensor;
        /** 逐元素反正切: arctan(self) */
        atan(out?: Tensor): Tensor;
        /** 逐元素二参数反正切: atan2(self, other)，返回 [-π, π] 弧度 */
        atan2(other: Tensor, out?: Tensor): Tensor;
        /** 逐元素反双曲正切: arctanh(self) */
        atanh(out?: Tensor): Tensor;
        /** 逐元素向上取整: ceil(self) */
        ceil(out?: Tensor): Tensor;
        /** 逐元素截断到 [min, max] 范围 */
        clamp(min?: number, max?: number, out?: Tensor): Tensor;
        /** 复数共轭: a+bi -> a-bi */
        conj(out?: Tensor): Tensor;
        /** 逐元素余弦: cos(self) */
        cos(out?: Tensor): Tensor;
        /** 逐元素双曲余弦: cosh(self) */
        cosh(out?: Tensor): Tensor;
        /** 逐元素除法: self / other (标量版本) */
        div(other: number | Tensor, roundingMode?: 'trunc' | 'floor', out?: Tensor): Tensor;
        /** 逐元素相等比较: self == other (标量版本) */
        eq(other: number | Tensor): Tensor;
        /** 逐元素误差函数: erf(self) */
        erf(out?: Tensor): Tensor;
        /** 逐元素互补误差函数: erfc(self) */
        erfc(out?: Tensor): Tensor;
        /** 逐元素指数: e^self */
        exp(out?: Tensor): Tensor;
        /** 逐元素 2 的幂: 2^self */
        exp2(out?: Tensor): Tensor;
        /** 逐元素 exp(self) - 1 */
        expm1(out?: Tensor): Tensor;
        /** 逐元素向下取整: floor(self) */
        floor(out?: Tensor): Tensor;
        /** 向下取整除法: floor(self / other) (标量版本) */
        floorDivide(other: number | Tensor, out?: Tensor): Tensor;
        /** C++ 风格取模: fmod(self, other) (标量版本) */
        fmod(other: number | Tensor, out?: Tensor): Tensor;
        /** 逐元素取小数部分: frac(self) */
        frac(out?: Tensor): Tensor;
        /** 逐元素大于等于比较: self >= other (标量版本) */
        ge(other: number | Tensor): Tensor;
        /** 逐元素大于比较: self > other (标量版本) */
        gt(other: number | Tensor): Tensor;
        /** 逐元素计算 0 阶修正贝塞尔函数: I₀(self) */
        i0(out?: Tensor): Tensor;
        /** 取虚部: a+bi -> b */
        imag(out?: Tensor): Tensor;
        /** 逐元素检查是否为有限数 */
        isfinite(): Tensor;
        /** 逐元素检查是否为无穷大 */
        isinf(): Tensor;
        /** 逐元素检查是否为 NaN */
        isnan(): Tensor;
        /** 逐元素小于等于比较: self <= other (标量版本) */
        le(other: number | Tensor): Tensor;
        /** 逐元素自然对数: ln(self) */
        log(out?: Tensor): Tensor;
        /** 逐元素以 10 为底对数: log10(self) */
        log10(out?: Tensor): Tensor;
        /** 逐元素 ln(1 + self) */
        log1p(out?: Tensor): Tensor;
        /** 逐元素以 2 为底对数: log2(self) */
        log2(out?: Tensor): Tensor;
        /** 逐元素逻辑非: !self */
        logicalNot(out?: Tensor): Tensor;
        /** 逐元素小于比较: self < other (标量版本) */
        lt(other: number | Tensor): Tensor;
        /** 逐元素最大值: max(self, other) */
        maximum(other: Tensor, out?: Tensor): Tensor;
        /** 逐元素最小值: min(self, other) */
        minimum(other: Tensor, out?: Tensor): Tensor;
        /** 逐元素乘法: self * other (标量版本) */
        mul(other: number | Tensor, out?: Tensor): Tensor;
        /** 逐元素不等比较: self != other (标量版本) */
        ne(other: number | Tensor): Tensor;
        /** 逐元素取反: -self */
        neg(out?: Tensor): Tensor;
        /** 逐元素幂运算: self ^ exponent (标量版本) */
        pow(exponent: number | Tensor, out?: Tensor): Tensor;
        /** 取实部: a+bi -> a */
        real(out?: Tensor): Tensor;
        /** 逐元素倒数: 1/self */
        reciprocal(out?: Tensor): Tensor;
        /** 逐元素 ReLU: max(0, self) */
        relu(): Tensor;
        /** Python 风格取模: remainder(self, other) (标量版本) */
        remainder(other: number | Tensor, out?: Tensor): Tensor;
        /** 逐元素四舍五入: round(self) */
        round(decimals?: number, out?: Tensor): Tensor;
        /** 逐元素平方根倒数: 1/sqrt(self) */
        rsqrt(out?: Tensor): Tensor;
        /** 逐元素 Sigmoid: 1 / (1 + exp(-self)) */
        sigmoid(out?: Tensor): Tensor;
        /** 逐元素符号: sign(self) */
        sign(out?: Tensor): Tensor;
        /** 逐元素正弦: sin(self) */
        sin(out?: Tensor): Tensor;
        /** 逐元素归一化 sinc: sin(πx)/(πx)，x=0 时返回 1 */
        sinc(out?: Tensor): Tensor;
        /** 逐元素双曲正弦: sinh(self) */
        sinh(out?: Tensor): Tensor;
        /** 逐元素平方根: sqrt(self) */
        sqrt(out?: Tensor): Tensor;
        /** 逐元素平方: self * self */
        square(out?: Tensor): Tensor;
        /** 逐元素减法: self - alpha * other (标量版本) */
        sub(other: number | Tensor, alpha?: number, out?: Tensor): Tensor;
        /** 逐元素正切: tan(self) */
        tan(out?: Tensor): Tensor;
        /** 逐元素双曲正切: tanh(self) */
        tanh(out?: Tensor): Tensor;
        /** 逐元素向零取整: trunc(self) */
        trunc(out?: Tensor): Tensor;
        /** 根据条件选择: condition ? self : other */
        where(condition: Tensor, other: Tensor): Tensor;
        dropout(p?: number, training?: boolean): Tensor;
        elu(alpha?: number): Tensor;
        gelu(approximate?: 'none' | 'tanh'): Tensor;
        hardtanh(minVal?: number, maxVal?: number): Tensor;
        leakyRelu(negativeSlope?: number): Tensor;
        logSoftmax(dim?: number, dtype?: DType): Tensor;
        /** LogSigmoid 激活: log(1 / (1 + exp(-self))) */
        logsigmoid(out?: Tensor): Tensor;
        /** SELU 激活 */
        selu(inplace?: boolean): Tensor;
        silu(): Tensor;
        softmax(dim?: number, dtype?: DType): Tensor;
        /** Softmin 激活: softmax(-self) */
        softmin(dim?: number, dtype?: DType): Tensor;
        /** 沿维度逻辑与 */
        all(dim?: number, keepdim?: boolean): Tensor;
        /** 沿维度逻辑或 */
        any(dim?: number, keepdim?: boolean): Tensor;
        /** 沿维度最大值索引 */
        argmax(dim?: number, keepdim?: boolean): Tensor;
        /** 沿维度最小值索引 */
        argmin(dim?: number, keepdim?: boolean): Tensor;
        /** 数值稳定的 log(sum(exp(x)))，公式: max(x) + log(sum(exp(x - max(x)))) */
        logsumexp(dim: number | number[], keepdim?: boolean): Tensor;
        /** 沿维度最大值及索引 */
        max(dim?: number, keepdim?: boolean): [Tensor, Tensor] | Tensor;
        /** 沿维度求均值 */
        mean(dim?: number | number[], keepdim?: boolean, dtype?: DType): Tensor;
        /** 沿维度最小值及索引 */
        min(dim?: number, keepdim?: boolean): [Tensor, Tensor] | Tensor;
        /** 沿维度求均值(忽略NaN) */
        nanmean(dim?: number | number[], keepdim?: boolean, dtype?: DType): Tensor;
        /** 沿维度求和(忽略NaN) */
        nansum(dim?: number | number[], keepdim?: boolean, dtype?: DType): Tensor;
        /** 沿维度范数 */
        norm(p?: number, dim?: number | number[], keepdim?: boolean): Tensor;
        /** 沿维度求积 */
        prod(dim?: number, keepdim?: boolean, dtype?: DType): Tensor;
        /** 沿维度标准差 */
        std(dim?: number | number[], correction?: number, keepdim?: boolean): Tensor;
        /** 沿维度求和 */
        sum(dim?: number | number[], keepdim?: boolean, dtype?: DType): Tensor;
        /** 沿维度方差 */
        variance(dim?: number | number[], correction?: number, keepdim?: boolean): Tensor;
        /** out = beta * self + alpha * (mat1 @ mat2) */
        addmm(mat1: Tensor, mat2: Tensor, beta?: number, alpha?: number, out?: Tensor): Tensor;
        /** out = beta * self + alpha * (mat @ vec) */
        addmv(mat: Tensor, vec: Tensor, beta?: number, alpha?: number, out?: Tensor): Tensor;
        /** out = beta * self + alpha * (batch1 @ batch2) */
        baddbmm(batch1: Tensor, batch2: Tensor, beta?: number, alpha?: number, out?: Tensor): Tensor;
        /** 批量矩阵乘法 */
        bmm(mat2: Tensor, out?: Tensor): Tensor;
        /** 向量点积 */
        dot(other: Tensor): Tensor;
        /** 矩阵乘法 (支持批量) */
        matmul(other: Tensor, out?: Tensor): Tensor;
        /** 2D 矩阵乘法 */
        mm(mat2: Tensor, out?: Tensor): Tensor;
        /** 矩阵-向量乘法 */
        mv(vec: Tensor, out?: Tensor): Tensor;
        /** 向量外积 */
        outer(vec2: Tensor, out?: Tensor): Tensor;
        /** 对角线构造 (1D->2D) 或提取 (2D->1D) */
        diag(diagonal?: number): Tensor;
        /** 获取对角线视图 (Partial View) */
        diagonal(offset?: number, dim1?: number, dim2?: number): Tensor;
        /** 矩阵迹: 对角线元素之和 */
        trace(): Tensor;
        /** 下三角矩阵: 保留 row >= col + diagonal 的元素，其余置零 */
        tril(diagonal?: number): Tensor;
        /** 上三角矩阵: 保留 row <= col + diagonal 的元素，其余置零 */
        triu(diagonal?: number): Tensor;
        /** Batch Normalization */
        batchNorm(runningMean?: Tensor, runningVar?: Tensor, weight?: Tensor, bias?: Tensor, training?: boolean, momentum?: number, eps?: number): Tensor;
        /** Group Normalization */
        groupNorm(numGroups: number, weight?: Tensor, bias?: Tensor, eps?: number): Tensor;
        /** Layer Normalization */
        layerNorm(normalizedShape: number[], weight?: Tensor, bias?: Tensor, eps?: number): Tensor;
        /** F.normalize: self / self.norm() */
        normalize(p?: number, dim?: number, eps?: number, out?: Tensor): Tensor;
        /** RMS Normalization */
        rmsNorm(normalizedShape: number[], weight?: Tensor, eps?: number): Tensor;
        /** 创建具有指定 size 和 stride 的视图（STFT 分帧等场景的核心操作） */
        asStrided(size: number[], stride: number[], storageOffset?: number): Tensor;
        /** 沿指定维度拼接张量序列 */
        cat(tensors: Tensor[], dim?: number): Tensor;
        /** 计算 N 阶前向差分: out[i] = input[i+1] - input[i] */
        diff(n?: number, dim?: number, prepend?: Tensor, append?: Tensor): Tensor;
        expand(size: number[]): Tensor;
        flatten(startDim?: number, endDim?: number): Tensor;
        /** 沿给定维度翻转张量元素顺序 */
        flip(dims: number | number[]): Tensor;
        /** 左右翻转 (沿 dim=1 翻转)，要求至少 2D */
        fliplr(): Tensor;
        /** 上下翻转 (沿 dim=0 翻转)，要求至少 1D */
        flipud(): Tensor;
        permute(dims: number | number[]): Tensor;
        /** 沿维度重复每个元素指定次数 */
        repeatInterleave(repeats: number | Tensor, dim?: number, outputSize?: number): Tensor;
        reshape(shape: number[]): Tensor;
        /** 沿指定维度选择单个索引，返回降维的视图 */
        select(dim: number, index: number): Tensor;
        /** 张量切片: 使用 Python 风格切片语法或整数索引返回视图 */
        slice(slices: string | number): Tensor;
        squeeze(dim?: number): Tensor;
        /** 沿新维度堆叠张量序列 */
        stack(tensors: Tensor[], dim?: number): Tensor;
        transpose(dim0: number, dim1: number): Tensor;
        unsqueeze(dim: number): Tensor;
        view(shape: number[]): Tensor;
        emptyLike(dtype?: DType, device?: string): Tensor;
        onesLike(dtype?: DType, device?: string): Tensor;
        zerosLike(dtype?: DType, device?: string): Tensor;
        /** 类型转换: 将 tensor 转换为指定的 dtype */
        cast(dtype: DType): Tensor;
        clone(): Tensor;
        /** 确保 tensor 按指定格式连续存储。如果已是目标格式则返回原 tensor。 */
        contiguous(memoryFormat?: unknown): Tensor;
        to(dtype?: DType, device?: string, copy?: boolean): Tensor;
        cummax(dim: number): [Tensor, Tensor];
        cummin(dim: number): [Tensor, Tensor];
        cumprod(dim: number, dtype?: DType): Tensor;
        cumsum(dim: number, dtype?: DType): Tensor;
        argsort(dim?: number, descending?: boolean, stable?: boolean): Tensor;
        sort(dim?: number, descending?: boolean, stable?: boolean): [Tensor, Tensor];
        topk(k: number, dim?: number, largest?: boolean, sorted?: boolean): [Tensor, Tensor];
        /** 自适应 2D 平均池化（自动计算 kernel/stride） */
        adaptiveAvgPool2d(outputSize: number | number[]): Tensor;
        /** 自适应 2D 最大池化 */
        adaptiveMaxPool2d(outputSize: number | number[], returnIndices?: boolean): Tensor;
        /** 1D 平均池化 */
        avgPool1d(kernelSize: number, stride?: number, padding?: number, ceilMode?: boolean, countIncludePad?: boolean): Tensor;
        /** 2D 平均池化 */
        avgPool2d(kernelSize: number | number[], stride?: number | number[], padding?: number | number[], ceilMode?: boolean, countIncludePad?: boolean, divisorOverride?: number): Tensor;
        /** 3D 平均池化 */
        avgPool3d(kernelSize: number | number[], stride?: number | number[], padding?: number | number[], ceilMode?: boolean, countIncludePad?: boolean, divisorOverride?: number): Tensor;
        /** 1D 卷积操作 */
        conv1d(weight: Tensor, bias?: Tensor, stride?: number, padding?: number | 'same' | 'valid', dilation?: number, groups?: number): Tensor;
        /** 2D 卷积操作 (PyTorch F.conv2d 兼容) */
        conv2d(weight: Tensor, bias?: Tensor, stride?: number | number[], padding?: number | number[] | 'same' | 'valid', dilation?: number | number[], groups?: number): Tensor;
        /** 3D 卷积操作 */
        conv3d(weight: Tensor, bias?: Tensor, stride?: number | number[], padding?: number | number[] | 'same' | 'valid', dilation?: number | number[], groups?: number): Tensor;
        /** 2D 转置卷积 (反卷积) */
        convTranspose2d(weight: Tensor, bias?: Tensor, stride?: number | number[], padding?: number | number[], outputPadding?: number | number[], groups?: number, dilation?: number | number[]): Tensor;
        /** 1D 最大池化 */
        maxPool1d(kernelSize: number, stride?: number, padding?: number, dilation?: number, ceilMode?: boolean, returnIndices?: boolean): Tensor;
        /** 2D 最大池化 */
        maxPool2d(kernelSize: number | number[], stride?: number | number[], padding?: number | number[], dilation?: number | number[], ceilMode?: boolean, returnIndices?: boolean): Tensor;
        /** 3D 最大池化 */
        maxPool3d(kernelSize: number | number[], stride?: number | number[], padding?: number | number[], dilation?: number | number[], ceilMode?: boolean, returnIndices?: boolean): Tensor;
        /** 沿维度选择索引: out[...] = self.select(dim, index[...]) */
        indexSelect(dim: number, index: Tensor, out?: Tensor): Tensor;
        /** 散射操作: out[index[...]][...] = src[...] */
        scatter(dim: number, index: Tensor, src: Tensor, out?: Tensor): Tensor;
        /** 散射加法: out[index[...]][...] += src[...] */
        scatterAdd(dim: number, index: Tensor, src: Tensor, out?: Tensor): Tensor;
        /** 通用散射归约 */
        scatterReduce(dim: number, index: Tensor, src: Tensor, reduce: 'sum' | 'prod' | 'mean' | 'amax' | 'amin', includeSelf?: boolean, out?: Tensor): Tensor;
    }

    // Static methods
    namespace Tensor {
        /** 从多项式分布中采样索引 */
        function multinomial(input: Tensor, numSamples: number, replacement?: boolean): Tensor;
    }
}

// ============================================================================
// Runtime Binding
// ============================================================================

Tensor.prototype.abs = function(out: any) {
    return ops.abs(this, out);
};

Tensor.prototype.acos = function(out: any) {
    return ops.acos(this, out);
};

Tensor.prototype.acosh = function(out: any) {
    return ops.acosh(this, out);
};

Tensor.prototype.add = function(other: any, alpha: any, out: any) {
    return ops.add(this, other, alpha);
};

Tensor.prototype.angle = function(out: any) {
    return ops.angle(this, out);
};

Tensor.prototype.asin = function(out: any) {
    return ops.asin(this, out);
};

Tensor.prototype.asinh = function(out: any) {
    return ops.asinh(this, out);
};

Tensor.prototype.atan = function(out: any) {
    return ops.atan(this, out);
};

Tensor.prototype.atan2 = function(other: any, out: any) {
    return ops.atan2(this, other, out);
};

Tensor.prototype.atanh = function(out: any) {
    return ops.atanh(this, out);
};

Tensor.prototype.ceil = function(out: any) {
    return ops.ceil(this, out);
};

Tensor.prototype.clamp = function(min: any, max: any, out: any) {
    return ops.clamp(this, min, max, out);
};

Tensor.prototype.conj = function(out: any) {
    return ops.conj(this, out);
};

Tensor.prototype.cos = function(out: any) {
    return ops.cos(this, out);
};

Tensor.prototype.cosh = function(out: any) {
    return ops.cosh(this, out);
};

Tensor.prototype.div = function(other: any, roundingMode: any, out: any) {
    return ops.div(this, other, roundingMode);
};

Tensor.prototype.eq = function(other: any) {
    return ops.eq(this, other);
};

Tensor.prototype.erf = function(out: any) {
    return ops.erf(this, out);
};

Tensor.prototype.erfc = function(out: any) {
    return ops.erfc(this, out);
};

Tensor.prototype.exp = function(out: any) {
    return ops.exp(this, out);
};

Tensor.prototype.exp2 = function(out: any) {
    return ops.exp2(this, out);
};

Tensor.prototype.expm1 = function(out: any) {
    return ops.expm1(this, out);
};

Tensor.prototype.floor = function(out: any) {
    return ops.floor(this, out);
};

Tensor.prototype.floorDivide = function(other: any, out: any) {
    return ops.floorDivide(this, other, out);
};

Tensor.prototype.fmod = function(other: any, out: any) {
    return ops.fmod(this, other, out);
};

Tensor.prototype.frac = function(out: any) {
    return ops.frac(this, out);
};

Tensor.prototype.ge = function(other: any) {
    return ops.ge(this, other);
};

Tensor.prototype.gt = function(other: any) {
    return ops.gt(this, other);
};

Tensor.prototype.i0 = function(out: any) {
    return ops.i0(this, out);
};

Tensor.prototype.imag = function(out: any) {
    return ops.imag(this, out);
};

Tensor.prototype.isfinite = function() {
    return ops.isfinite(this);
};

Tensor.prototype.isinf = function() {
    return ops.isinf(this);
};

Tensor.prototype.isnan = function() {
    return ops.isnan(this);
};

Tensor.prototype.le = function(other: any) {
    return ops.le(this, other);
};

Tensor.prototype.log = function(out: any) {
    return ops.log(this, out);
};

Tensor.prototype.log10 = function(out: any) {
    return ops.log10(this, out);
};

Tensor.prototype.log1p = function(out: any) {
    return ops.log1p(this, out);
};

Tensor.prototype.log2 = function(out: any) {
    return ops.log2(this, out);
};

Tensor.prototype.logicalNot = function(out: any) {
    return ops.logicalNot(this, out);
};

Tensor.prototype.lt = function(other: any) {
    return ops.lt(this, other);
};

Tensor.prototype.maximum = function(other: any, out: any) {
    return ops.maximum(this, other, out);
};

Tensor.prototype.minimum = function(other: any, out: any) {
    return ops.minimum(this, other, out);
};

Tensor.prototype.mul = function(other: any, out: any) {
    return ops.mul(this, other);
};

Tensor.prototype.ne = function(other: any) {
    return ops.ne(this, other);
};

Tensor.prototype.neg = function(out: any) {
    return ops.neg(this, out);
};

Tensor.prototype.pow = function(exponent: any, out: any) {
    return ops.pow(this, exponent, out);
};

Tensor.prototype.real = function(out: any) {
    return ops.real(this, out);
};

Tensor.prototype.reciprocal = function(out: any) {
    return ops.reciprocal(this, out);
};

Tensor.prototype.relu = function() {
    return ops.relu(this);
};

Tensor.prototype.remainder = function(other: any, out: any) {
    return ops.remainder(this, other, out);
};

Tensor.prototype.round = function(decimals: any, out: any) {
    return ops.round(this, decimals, out);
};

Tensor.prototype.rsqrt = function(out: any) {
    return ops.rsqrt(this, out);
};

Tensor.prototype.sigmoid = function(out: any) {
    return ops.sigmoid(this, out);
};

Tensor.prototype.sign = function(out: any) {
    return ops.sign(this, out);
};

Tensor.prototype.sin = function(out: any) {
    return ops.sin(this, out);
};

Tensor.prototype.sinc = function(out: any) {
    return ops.sinc(this, out);
};

Tensor.prototype.sinh = function(out: any) {
    return ops.sinh(this, out);
};

Tensor.prototype.sqrt = function(out: any) {
    return ops.sqrt(this, out);
};

Tensor.prototype.square = function(out: any) {
    return ops.square(this, out);
};

Tensor.prototype.sub = function(other: any, alpha: any, out: any) {
    return ops.sub(this, other, alpha);
};

Tensor.prototype.tan = function(out: any) {
    return ops.tan(this, out);
};

Tensor.prototype.tanh = function(out: any) {
    return ops.tanh(this, out);
};

Tensor.prototype.trunc = function(out: any) {
    return ops.trunc(this, out);
};

Tensor.prototype.where = function(condition: any, other: any) {
    return ops.where(condition, this, other);
};

Tensor.prototype.dropout = function(p: any, training: any) {
    return ops.dropout(this, p, training);
};

Tensor.prototype.elu = function(alpha: any) {
    return ops.elu(this, alpha);
};

Tensor.prototype.gelu = function(approximate: any) {
    return ops.gelu(this, approximate);
};

Tensor.prototype.hardtanh = function(minVal: any, maxVal: any) {
    return ops.hardtanh(this, minVal, maxVal);
};

Tensor.prototype.leakyRelu = function(negativeSlope: any) {
    return ops.leakyRelu(this, negativeSlope);
};

Tensor.prototype.logSoftmax = function(dim: any, dtype: any) {
    return ops.logSoftmax(this, dim, dtype);
};

Tensor.prototype.logsigmoid = function(out: any) {
    return ops.logsigmoid(this, out);
};

Tensor.prototype.selu = function(inplace: any) {
    return ops.selu(this, inplace);
};

Tensor.prototype.silu = function() {
    return ops.silu(this);
};

Tensor.prototype.softmax = function(dim: any, dtype: any) {
    return ops.softmax(this, dim, dtype);
};

Tensor.prototype.softmin = function(dim: any, dtype: any) {
    return ops.softmin(this, dim, dtype);
};

Tensor.prototype.all = function(dim: any, keepdim: any) {
    return ops.all(this, dim, keepdim);
};

Tensor.prototype.any = function(dim: any, keepdim: any) {
    return ops.any(this, dim, keepdim);
};

Tensor.prototype.argmax = function(dim: any, keepdim: any) {
    return ops.argmax(this, dim, keepdim);
};

Tensor.prototype.argmin = function(dim: any, keepdim: any) {
    return ops.argmin(this, dim, keepdim);
};

Tensor.prototype.logsumexp = function(dim: any, keepdim: any) {
    return ops.logsumexp(this, dim, keepdim);
};

Tensor.prototype.max = function(dim: any, keepdim: any) {
    return ops.max(this, dim, keepdim);
};

Tensor.prototype.mean = function(dim: any, keepdim: any, dtype: any) {
    return ops.mean(this, dim, keepdim, dtype);
};

Tensor.prototype.min = function(dim: any, keepdim: any) {
    return ops.min(this, dim, keepdim);
};

Tensor.prototype.nanmean = function(dim: any, keepdim: any, dtype: any) {
    return ops.nanmean(this, dim, keepdim, dtype);
};

Tensor.prototype.nansum = function(dim: any, keepdim: any, dtype: any) {
    return ops.nansum(this, dim, keepdim, dtype);
};

Tensor.prototype.norm = function(p: any, dim: any, keepdim: any) {
    return ops.norm(this, p, dim, keepdim);
};

Tensor.prototype.prod = function(dim: any, keepdim: any, dtype: any) {
    return ops.prod(this, dim, keepdim, dtype);
};

Tensor.prototype.std = function(dim: any, correction: any, keepdim: any) {
    return ops.std(this, dim, correction, keepdim);
};

Tensor.prototype.sum = function(dim: any, keepdim: any, dtype: any) {
    return ops.sum(this, dim, keepdim, dtype);
};

Tensor.prototype.variance = function(dim: any, correction: any, keepdim: any) {
    return ops.variance(this, dim, correction, keepdim);
};

Tensor.prototype.addmm = function(mat1: any, mat2: any, beta: any, alpha: any, out: any) {
    return ops.addmm(this, mat1, mat2, beta, alpha, out);
};

Tensor.prototype.addmv = function(mat: any, vec: any, beta: any, alpha: any, out: any) {
    return ops.addmv(this, mat, vec, beta, alpha, out);
};

Tensor.prototype.baddbmm = function(batch1: any, batch2: any, beta: any, alpha: any, out: any) {
    return ops.baddbmm(this, batch1, batch2, beta, alpha, out);
};

Tensor.prototype.bmm = function(mat2: any, out: any) {
    return ops.bmm(this, mat2, out);
};

Tensor.prototype.dot = function(other: any) {
    return ops.dot(this, other);
};

Tensor.prototype.matmul = function(other: any, out: any) {
    return ops.matmul(this, other, out);
};

Tensor.prototype.mm = function(mat2: any, out: any) {
    return ops.mm(this, mat2, out);
};

Tensor.prototype.mv = function(vec: any, out: any) {
    return ops.mv(this, vec, out);
};

Tensor.prototype.outer = function(vec2: any, out: any) {
    return ops.outer(this, vec2, out);
};

Tensor.prototype.diag = function(diagonal: any) {
    return ops.diag(this, diagonal);
};

Tensor.prototype.diagonal = function(offset: any, dim1: any, dim2: any) {
    return ops.diagonal(this, offset, dim1, dim2);
};

Tensor.prototype.trace = function() {
    return ops.trace(this);
};

Tensor.prototype.tril = function(diagonal: any) {
    return ops.tril(this, diagonal);
};

Tensor.prototype.triu = function(diagonal: any) {
    return ops.triu(this, diagonal);
};

Tensor.prototype.batchNorm = function(runningMean: any, runningVar: any, weight: any, bias: any, training: any, momentum: any, eps: any) {
    return ops.batchNorm(this, runningMean, runningVar, weight, bias, training, momentum, eps);
};

Tensor.prototype.groupNorm = function(numGroups: any, weight: any, bias: any, eps: any) {
    return ops.groupNorm(this, numGroups, weight, bias, eps);
};

Tensor.prototype.layerNorm = function(normalizedShape: any, weight: any, bias: any, eps: any) {
    return ops.layerNorm(this, normalizedShape, weight, bias, eps);
};

Tensor.prototype.normalize = function(p: any, dim: any, eps: any, out: any) {
    return ops.normalize(this, p, dim, eps, out);
};

Tensor.prototype.rmsNorm = function(normalizedShape: any, weight: any, eps: any) {
    return ops.rmsNorm(this, normalizedShape, weight, eps);
};

Tensor.prototype.asStrided = function(size: any, stride: any, storageOffset: any) {
    return ops.asStrided(this, size, stride, storageOffset);
};

Tensor.prototype.cat = function(tensors: any, dim: any) {
    return ops.cat(tensors, dim, this);
};

Tensor.prototype.diff = function(n: any, dim: any, prepend: any, append: any) {
    return ops.diff(this, n, dim, prepend, append);
};

Tensor.prototype.expand = function(size: any) {
    return ops.expand(this, size);
};

Tensor.prototype.flatten = function(startDim: any, endDim: any) {
    return ops.flatten(this, startDim, endDim);
};

Tensor.prototype.flip = function(dims: any) {
    return ops.flip(this, dims);
};

Tensor.prototype.fliplr = function() {
    return ops.fliplr(this);
};

Tensor.prototype.flipud = function() {
    return ops.flipud(this);
};

Tensor.prototype.permute = function(dims: any) {
    return ops.permute(this, dims);
};

Tensor.prototype.repeatInterleave = function(repeats: any, dim: any, outputSize: any) {
    return ops.repeatInterleave(this, repeats, dim, outputSize);
};

Tensor.prototype.reshape = function(shape: any) {
    return ops.reshape(this, shape);
};

Tensor.prototype.select = function(dim: any, index: any) {
    return ops.select(this, dim, index);
};

Tensor.prototype.slice = function(slices: any) {
    return ops.slice(this, slices);
};

Tensor.prototype.squeeze = function(dim: any) {
    return ops.squeeze(this, dim);
};

Tensor.prototype.stack = function(tensors: any, dim: any) {
    return ops.stack(tensors, dim, this);
};

Tensor.prototype.transpose = function(dim0: any, dim1: any) {
    return ops.transpose(this, dim0, dim1);
};

Tensor.prototype.unsqueeze = function(dim: any) {
    return ops.unsqueeze(this, dim);
};

Tensor.prototype.view = function(shape: any) {
    return ops.view(this, shape);
};

Tensor.prototype.emptyLike = function(dtype: any, device: any) {
    return ops.emptyLike(this, dtype, device);
};

Tensor.prototype.onesLike = function(dtype: any, device: any) {
    return ops.onesLike(this, dtype, device);
};

Tensor.prototype.zerosLike = function(dtype: any, device: any) {
    return ops.zerosLike(this, dtype, device);
};

Tensor.prototype.cast = function(dtype: any) {
    return ops.cast(this, dtype);
};

Tensor.prototype.clone = function() {
    return ops.clone(this);
};

Tensor.prototype.contiguous = function(memoryFormat: any) {
    return ops.contiguous(this, memoryFormat);
};

Tensor.prototype.to = function(dtype: any, device: any, copy: any) {
    return ops.to(this, dtype, device, copy);
};

Tensor.prototype.cummax = function(dim: any) {
    return ops.cummax(this, dim);
};

Tensor.prototype.cummin = function(dim: any) {
    return ops.cummin(this, dim);
};

Tensor.prototype.cumprod = function(dim: any, dtype: any) {
    return ops.cumprod(this, dim, dtype);
};

Tensor.prototype.cumsum = function(dim: any, dtype: any) {
    return ops.cumsum(this, dim, dtype);
};

Tensor.prototype.argsort = function(dim: any, descending: any, stable: any) {
    return ops.argsort(this, dim, descending, stable);
};

Tensor.prototype.sort = function(dim: any, descending: any, stable: any) {
    return ops.sort(this, dim, descending, stable);
};

Tensor.prototype.topk = function(k: any, dim: any, largest: any, sorted: any) {
    return ops.topk(this, k, dim, largest, sorted);
};

Tensor.prototype.adaptiveAvgPool2d = function(outputSize: any) {
    return ops.adaptiveAvgPool2d(this, outputSize);
};

Tensor.prototype.adaptiveMaxPool2d = function(outputSize: any, returnIndices: any) {
    return ops.adaptiveMaxPool2d(this, outputSize, returnIndices);
};

Tensor.prototype.avgPool1d = function(kernelSize: any, stride: any, padding: any, ceilMode: any, countIncludePad: any) {
    return ops.avgPool1d(this, kernelSize, stride, padding, ceilMode, countIncludePad);
};

Tensor.prototype.avgPool2d = function(kernelSize: any, stride: any, padding: any, ceilMode: any, countIncludePad: any, divisorOverride: any) {
    return ops.avgPool2d(this, kernelSize, stride, padding, ceilMode, countIncludePad, divisorOverride);
};

Tensor.prototype.avgPool3d = function(kernelSize: any, stride: any, padding: any, ceilMode: any, countIncludePad: any, divisorOverride: any) {
    return ops.avgPool3d(this, kernelSize, stride, padding, ceilMode, countIncludePad, divisorOverride);
};

Tensor.prototype.conv1d = function(weight: any, bias: any, stride: any, padding: any, dilation: any, groups: any) {
    return ops.conv1d(this, weight, bias, stride, padding, dilation, groups);
};

Tensor.prototype.conv2d = function(weight: any, bias: any, stride: any, padding: any, dilation: any, groups: any) {
    return ops.conv2d(this, weight, bias, stride, padding, dilation, groups);
};

Tensor.prototype.conv3d = function(weight: any, bias: any, stride: any, padding: any, dilation: any, groups: any) {
    return ops.conv3d(this, weight, bias, stride, padding, dilation, groups);
};

Tensor.prototype.convTranspose2d = function(weight: any, bias: any, stride: any, padding: any, outputPadding: any, groups: any, dilation: any) {
    return ops.convTranspose2d(this, weight, bias, stride, padding, outputPadding, groups, dilation);
};

Tensor.prototype.maxPool1d = function(kernelSize: any, stride: any, padding: any, dilation: any, ceilMode: any, returnIndices: any) {
    return ops.maxPool1d(this, kernelSize, stride, padding, dilation, ceilMode, returnIndices);
};

Tensor.prototype.maxPool2d = function(kernelSize: any, stride: any, padding: any, dilation: any, ceilMode: any, returnIndices: any) {
    return ops.maxPool2d(this, kernelSize, stride, padding, dilation, ceilMode, returnIndices);
};

Tensor.prototype.maxPool3d = function(kernelSize: any, stride: any, padding: any, dilation: any, ceilMode: any, returnIndices: any) {
    return ops.maxPool3d(this, kernelSize, stride, padding, dilation, ceilMode, returnIndices);
};

Tensor.prototype.indexSelect = function(dim: any, index: any, out: any) {
    return ops.indexSelect(this, dim, index, out);
};

Tensor.prototype.scatter = function(dim: any, index: any, src: any, out: any) {
    return ops.scatter(this, dim, index, src, out);
};

Tensor.prototype.scatterAdd = function(dim: any, index: any, src: any, out: any) {
    return ops.scatterAdd(this, dim, index, src, out);
};

Tensor.prototype.scatterReduce = function(dim: any, index: any, src: any, reduce: any, includeSelf: any, out: any) {
    return ops.scatterReduce(this, dim, index, src, reduce, includeSelf, out);
};

(Tensor as any).multinomial = function(input: any, numSamples: any, replacement: any) {
    return ops.multinomial(input, numSamples, replacement);
};
