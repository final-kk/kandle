/**
 * v5 Generated Type Declarations
 * DO NOT EDIT - Generated from OpRegistry
 */

import type { DType } from '@kandle/types';
import type { Tensor } from '../tensor';

// Operator function signatures
export interface Ops {
    abs(self: Tensor, out?: Tensor): Tensor;
    acos(self: Tensor, out?: Tensor): Tensor;
    acosh(self: Tensor, out?: Tensor): Tensor;
    add: {
        (self: Tensor, other: number, alpha?: number): Tensor;
        (self: Tensor, other: Tensor, alpha?: number, out?: Tensor): Tensor;
    };
    angle(self: Tensor, out?: Tensor): Tensor;
    asin(self: Tensor, out?: Tensor): Tensor;
    asinh(self: Tensor, out?: Tensor): Tensor;
    atan(self: Tensor, out?: Tensor): Tensor;
    atan2(self: Tensor, other: Tensor, out?: Tensor): Tensor;
    atanh(self: Tensor, out?: Tensor): Tensor;
    ceil(self: Tensor, out?: Tensor): Tensor;
    clamp(self: Tensor, min?: number, max?: number, out?: Tensor): Tensor;
    conj(self: Tensor, out?: Tensor): Tensor;
    cos(self: Tensor, out?: Tensor): Tensor;
    cosh(self: Tensor, out?: Tensor): Tensor;
    div: {
        (self: Tensor, other: number, roundingMode?: 'trunc' | 'floor'): Tensor;
        (self: Tensor, other: Tensor, roundingMode?: 'trunc' | 'floor', out?: Tensor): Tensor;
    };
    eq: {
        (self: Tensor, other: number): Tensor;
        (self: Tensor, other: Tensor): Tensor;
    };
    erf(self: Tensor, out?: Tensor): Tensor;
    erfc(self: Tensor, out?: Tensor): Tensor;
    exp(self: Tensor, out?: Tensor): Tensor;
    exp2(self: Tensor, out?: Tensor): Tensor;
    expm1(self: Tensor, out?: Tensor): Tensor;
    floor(self: Tensor, out?: Tensor): Tensor;
    floorDivide: {
        (self: Tensor, other: number, out?: Tensor): Tensor;
        (self: Tensor, other: Tensor, out?: Tensor): Tensor;
    };
    fmod: {
        (self: Tensor, other: number, out?: Tensor): Tensor;
        (self: Tensor, other: Tensor, out?: Tensor): Tensor;
    };
    frac(self: Tensor, out?: Tensor): Tensor;
    ge: {
        (self: Tensor, other: number): Tensor;
        (self: Tensor, other: Tensor): Tensor;
    };
    gt: {
        (self: Tensor, other: number): Tensor;
        (self: Tensor, other: Tensor): Tensor;
    };
    i0(self: Tensor, out?: Tensor): Tensor;
    imag(self: Tensor, out?: Tensor): Tensor;
    isfinite(self: Tensor): Tensor;
    isinf(self: Tensor): Tensor;
    isnan(self: Tensor): Tensor;
    le: {
        (self: Tensor, other: number): Tensor;
        (self: Tensor, other: Tensor): Tensor;
    };
    log(self: Tensor, out?: Tensor): Tensor;
    log10(self: Tensor, out?: Tensor): Tensor;
    log1p(self: Tensor, out?: Tensor): Tensor;
    log2(self: Tensor, out?: Tensor): Tensor;
    logicalNot(self: Tensor, out?: Tensor): Tensor;
    lt: {
        (self: Tensor, other: number): Tensor;
        (self: Tensor, other: Tensor): Tensor;
    };
    maximum(self: Tensor, other: Tensor, out?: Tensor): Tensor;
    minimum(self: Tensor, other: Tensor, out?: Tensor): Tensor;
    mul: {
        (self: Tensor, other: number): Tensor;
        (self: Tensor, other: Tensor, out?: Tensor): Tensor;
    };
    ne: {
        (self: Tensor, other: number): Tensor;
        (self: Tensor, other: Tensor): Tensor;
    };
    neg(self: Tensor, out?: Tensor): Tensor;
    pow: {
        (self: Tensor, exponent: number, out?: Tensor): Tensor;
        (self: Tensor, exponent: Tensor, out?: Tensor): Tensor;
    };
    real(self: Tensor, out?: Tensor): Tensor;
    reciprocal(self: Tensor, out?: Tensor): Tensor;
    relu(self: Tensor): Tensor;
    remainder: {
        (self: Tensor, other: number, out?: Tensor): Tensor;
        (self: Tensor, other: Tensor, out?: Tensor): Tensor;
    };
    round(self: Tensor, decimals?: number, out?: Tensor): Tensor;
    rsqrt(self: Tensor, out?: Tensor): Tensor;
    sigmoid(self: Tensor, out?: Tensor): Tensor;
    sign(self: Tensor, out?: Tensor): Tensor;
    sin(self: Tensor, out?: Tensor): Tensor;
    sinc(self: Tensor, out?: Tensor): Tensor;
    sinh(self: Tensor, out?: Tensor): Tensor;
    sqrt(self: Tensor, out?: Tensor): Tensor;
    square(self: Tensor, out?: Tensor): Tensor;
    sub: {
        (self: Tensor, other: number, alpha?: number): Tensor;
        (self: Tensor, other: Tensor, alpha?: number, out?: Tensor): Tensor;
    };
    tan(self: Tensor, out?: Tensor): Tensor;
    tanh(self: Tensor, out?: Tensor): Tensor;
    trunc(self: Tensor, out?: Tensor): Tensor;
    where(condition: Tensor, self: Tensor, other: Tensor): Tensor;
    dropout(self: Tensor, p?: number, training?: boolean): Tensor;
    elu(self: Tensor, alpha?: number): Tensor;
    gelu(self: Tensor, approximate?: 'none' | 'tanh'): Tensor;
    hardtanh(self: Tensor, minVal?: number, maxVal?: number): Tensor;
    leakyRelu(self: Tensor, negativeSlope?: number): Tensor;
    logSoftmax(self: Tensor, dim?: number, dtype?: DType): Tensor;
    logsigmoid(self: Tensor, out?: Tensor): Tensor;
    selu(self: Tensor, inplace?: boolean): Tensor;
    silu(self: Tensor): Tensor;
    softmax(self: Tensor, dim?: number, dtype?: DType): Tensor;
    softmin(self: Tensor, dim?: number, dtype?: DType): Tensor;
    all(self: Tensor, dim?: number, keepdim?: boolean): Tensor;
    any(self: Tensor, dim?: number, keepdim?: boolean): Tensor;
    argmax(self: Tensor, dim?: number, keepdim?: boolean): Tensor;
    argmin(self: Tensor, dim?: number, keepdim?: boolean): Tensor;
    logsumexp(self: Tensor, dim: number | number[], keepdim?: boolean): Tensor;
    max: {
        (self: Tensor, dim: number, keepdim?: boolean): [Tensor, Tensor];
        (self: Tensor): Tensor;
    };
    mean(self: Tensor, dim?: number | number[], keepdim?: boolean, dtype?: DType): Tensor;
    min: {
        (self: Tensor, dim: number, keepdim?: boolean): [Tensor, Tensor];
        (self: Tensor): Tensor;
    };
    nanmean(self: Tensor, dim?: number | number[], keepdim?: boolean, dtype?: DType): Tensor;
    nansum(self: Tensor, dim?: number | number[], keepdim?: boolean, dtype?: DType): Tensor;
    norm(self: Tensor, p?: number, dim?: number | number[], keepdim?: boolean): Tensor;
    prod(self: Tensor, dim?: number, keepdim?: boolean, dtype?: DType): Tensor;
    std(self: Tensor, dim?: number | number[], correction?: number, keepdim?: boolean): Tensor;
    sum(self: Tensor, dim?: number | number[], keepdim?: boolean, dtype?: DType): Tensor;
    variance(self: Tensor, dim?: number | number[], correction?: number, keepdim?: boolean): Tensor;
    addmm(self: Tensor, mat1: Tensor, mat2: Tensor, beta?: number, alpha?: number, out?: Tensor): Tensor;
    addmv(self: Tensor, mat: Tensor, vec: Tensor, beta?: number, alpha?: number, out?: Tensor): Tensor;
    baddbmm(self: Tensor, batch1: Tensor, batch2: Tensor, beta?: number, alpha?: number, out?: Tensor): Tensor;
    bmm(self: Tensor, mat2: Tensor, out?: Tensor): Tensor;
    dot(self: Tensor, other: Tensor): Tensor;
    linear(input: Tensor, weight: Tensor, bias?: Tensor): Tensor;
    matmul(self: Tensor, other: Tensor, out?: Tensor): Tensor;
    mm(self: Tensor, mat2: Tensor, out?: Tensor): Tensor;
    mv(self: Tensor, vec: Tensor, out?: Tensor): Tensor;
    outer(self: Tensor, vec2: Tensor, out?: Tensor): Tensor;
    diag(self: Tensor, diagonal?: number): Tensor;
    diagonal(self: Tensor, offset?: number, dim1?: number, dim2?: number): Tensor;
    trace(self: Tensor): Tensor;
    tril(self: Tensor, diagonal?: number): Tensor;
    triu(self: Tensor, diagonal?: number): Tensor;
    batchNorm(self: Tensor, runningMean?: Tensor, runningVar?: Tensor, weight?: Tensor, bias?: Tensor, training?: boolean, momentum?: number, eps?: number): Tensor;
    groupNorm(self: Tensor, numGroups: number, weight?: Tensor, bias?: Tensor, eps?: number): Tensor;
    layerNorm(self: Tensor, normalizedShape: number[], weight?: Tensor, bias?: Tensor, eps?: number): Tensor;
    normalize(self: Tensor, p?: number, dim?: number, eps?: number, out?: Tensor): Tensor;
    rmsNorm(self: Tensor, normalizedShape: number[], weight?: Tensor, eps?: number): Tensor;
    asStrided(self: Tensor, size: number[], stride: number[], storageOffset?: number): Tensor;
    cat(tensors: Tensor[], dim?: number, out?: Tensor): Tensor;
    diff(self: Tensor, n?: number, dim?: number, prepend?: Tensor, append?: Tensor): Tensor;
    expand(self: Tensor, size: number[]): Tensor;
    flatten(self: Tensor, startDim?: number, endDim?: number): Tensor;
    flip(self: Tensor, dims: number | number[]): Tensor;
    fliplr(self: Tensor): Tensor;
    flipud(self: Tensor): Tensor;
    permute(self: Tensor, dims: number | number[]): Tensor;
    repeatInterleave(self: Tensor, repeats: number | Tensor, dim?: number, outputSize?: number): Tensor;
    reshape(self: Tensor, shape: number[]): Tensor;
    select(self: Tensor, dim: number, index: number): Tensor;
    slice(self: Tensor, slices: string | number): Tensor;
    squeeze(self: Tensor, dim?: number): Tensor;
    stack(tensors: Tensor[], dim?: number, out?: Tensor): Tensor;
    transpose(self: Tensor, dim0: number, dim1: number): Tensor;
    unsqueeze(self: Tensor, dim: number): Tensor;
    view(self: Tensor, shape: number[]): Tensor;
    arange(start: number, end?: number, step?: number, dtype?: DType, device?: string): Tensor;
    bartlettWindow(windowLength: number, periodic?: boolean, dtype?: DType, device?: string): Tensor;
    blackmanWindow(windowLength: number, periodic?: boolean, dtype?: DType, device?: string): Tensor;
    empty(size: number[], dtype?: DType, device?: string): Tensor;
    emptyLike(self: Tensor, dtype?: DType, device?: string): Tensor;
    eye(n: number, m?: number, dtype?: DType, device?: string): Tensor;
    full(size: number[], fillValue: number, dtype?: DType, device?: string): Tensor;
    hammingWindow(windowLength: number, periodic?: boolean, alpha?: number, beta?: number, dtype?: DType, device?: string): Tensor;
    hannWindow(windowLength: number, periodic?: boolean, dtype?: DType, device?: string): Tensor;
    kaiserWindow(windowLength: number, periodic?: boolean, beta?: number, dtype?: DType, device?: string): Tensor;
    linspace(start: number, end: number, steps: number, dtype?: DType, device?: string): Tensor;
    multinomial(input: Tensor, numSamples: number, replacement?: boolean): Tensor;
    ones(size: number[], dtype?: DType, device?: string): Tensor;
    onesLike(self: Tensor, dtype?: DType, device?: string): Tensor;
    pad(input: Tensor, pad: number[], mode?: 'constant' | 'reflect' | 'replicate' | 'circular', value?: number): Tensor;
    rand(size: number[], dtype?: DType, device?: string): Tensor;
    randint(low: number, high: number, size: number[], dtype?: DType, device?: string): Tensor;
    randn(size: number[], dtype?: DType, device?: string): Tensor;
    zeros(size: number[], dtype?: DType, device?: string): Tensor;
    zerosLike(self: Tensor, dtype?: DType, device?: string): Tensor;
    cast(self: Tensor, dtype: DType): Tensor;
    clone(self: Tensor): Tensor;
    contiguous(self: Tensor, memoryFormat?: unknown): Tensor;
    to(self: Tensor, dtype?: DType, device?: string, copy?: boolean): Tensor;
    cummax(self: Tensor, dim: number): [Tensor, Tensor];
    cummin(self: Tensor, dim: number): [Tensor, Tensor];
    cumprod(self: Tensor, dim: number, dtype?: DType): Tensor;
    cumsum(self: Tensor, dim: number, dtype?: DType): Tensor;
    argsort(self: Tensor, dim?: number, descending?: boolean, stable?: boolean): Tensor;
    sort(self: Tensor, dim?: number, descending?: boolean, stable?: boolean): [Tensor, Tensor];
    topk(self: Tensor, k: number, dim?: number, largest?: boolean, sorted?: boolean): [Tensor, Tensor];
    adaptiveAvgPool2d(input: Tensor, outputSize: number | number[]): Tensor;
    adaptiveMaxPool2d(input: Tensor, outputSize: number | number[], returnIndices?: boolean): Tensor;
    avgPool1d(input: Tensor, kernelSize: number, stride?: number, padding?: number, ceilMode?: boolean, countIncludePad?: boolean): Tensor;
    avgPool2d(input: Tensor, kernelSize: number | number[], stride?: number | number[], padding?: number | number[], ceilMode?: boolean, countIncludePad?: boolean, divisorOverride?: number): Tensor;
    avgPool3d(input: Tensor, kernelSize: number | number[], stride?: number | number[], padding?: number | number[], ceilMode?: boolean, countIncludePad?: boolean, divisorOverride?: number): Tensor;
    conv1d(input: Tensor, weight: Tensor, bias?: Tensor, stride?: number, padding?: number | 'same' | 'valid', dilation?: number, groups?: number): Tensor;
    conv2d(input: Tensor, weight: Tensor, bias?: Tensor, stride?: number | number[], padding?: number | number[] | 'same' | 'valid', dilation?: number | number[], groups?: number): Tensor;
    conv3d(input: Tensor, weight: Tensor, bias?: Tensor, stride?: number | number[], padding?: number | number[] | 'same' | 'valid', dilation?: number | number[], groups?: number): Tensor;
    convTranspose2d(input: Tensor, weight: Tensor, bias?: Tensor, stride?: number | number[], padding?: number | number[], outputPadding?: number | number[], groups?: number, dilation?: number | number[]): Tensor;
    maxPool1d(input: Tensor, kernelSize: number, stride?: number, padding?: number, dilation?: number, ceilMode?: boolean, returnIndices?: boolean): Tensor;
    maxPool2d(input: Tensor, kernelSize: number | number[], stride?: number | number[], padding?: number | number[], dilation?: number | number[], ceilMode?: boolean, returnIndices?: boolean): Tensor;
    maxPool3d(input: Tensor, kernelSize: number | number[], stride?: number | number[], padding?: number | number[], dilation?: number | number[], ceilMode?: boolean, returnIndices?: boolean): Tensor;
    embedding(input: Tensor, weight: Tensor, paddingIdx?: number, maxNorm?: number, normType?: number, scaleGradByFreq?: boolean, sparse?: boolean): Tensor;
    indexSelect(self: Tensor, dim: number, index: Tensor, out?: Tensor): Tensor;
    scatter(self: Tensor, dim: number, index: Tensor, src: Tensor, out?: Tensor): Tensor;
    scatterAdd(self: Tensor, dim: number, index: Tensor, src: Tensor, out?: Tensor): Tensor;
    scatterReduce(self: Tensor, dim: number, index: Tensor, src: Tensor, reduce: 'sum' | 'prod' | 'mean' | 'amax' | 'amin', includeSelf?: boolean, out?: Tensor): Tensor;
    copy_(self: Tensor, src: Tensor): Tensor;
    scaledDotProductAttention(query: Tensor, key: Tensor, value: Tensor, attnMask?: Tensor, dropoutP?: number, isCausal?: boolean, scale?: number): Tensor;
    fft(input: Tensor, n?: number, dim?: number, norm?: 'forward' | 'backward' | 'ortho'): Tensor;
    fft2(input: Tensor, s?: number[], dim?: number | number[], norm?: 'forward' | 'backward' | 'ortho'): Tensor;
    fftfreq(n: number, d?: number): Tensor;
    fftn(input: Tensor, s?: number[], dim?: number | number[], norm?: 'forward' | 'backward' | 'ortho'): Tensor;
    fftshift(input: Tensor, dim?: number | number[]): Tensor;
    hfft(input: Tensor, n?: number, dim?: number, norm?: 'forward' | 'backward' | 'ortho'): Tensor;
    ifft(input: Tensor, n?: number, dim?: number, norm?: 'forward' | 'backward' | 'ortho'): Tensor;
    ifft2(input: Tensor, s?: number[], dim?: number | number[], norm?: 'forward' | 'backward' | 'ortho'): Tensor;
    ifftn(input: Tensor, s?: number[], dim?: number | number[], norm?: 'forward' | 'backward' | 'ortho'): Tensor;
    ifftshift(input: Tensor, dim?: number | number[]): Tensor;
    ihfft(input: Tensor, n?: number, dim?: number, norm?: 'forward' | 'backward' | 'ortho'): Tensor;
    irfft(input: Tensor, n?: number, dim?: number, norm?: 'forward' | 'backward' | 'ortho'): Tensor;
    irfft2(input: Tensor, s?: number[], dim?: number | number[], norm?: 'forward' | 'backward' | 'ortho'): Tensor;
    irfftn(input: Tensor, s?: number[], dim?: number | number[], norm?: 'forward' | 'backward' | 'ortho'): Tensor;
    istft(input: Tensor, n_fft: number, hop_length?: number, win_length?: number, window?: Tensor, center?: boolean, normalized?: boolean, onesided?: boolean, length?: number, return_complex?: boolean): Tensor;
    rfft(input: Tensor, n?: number, dim?: number, norm?: 'forward' | 'backward' | 'ortho'): Tensor;
    rfft2(input: Tensor, s?: number[], dim?: number | number[], norm?: 'forward' | 'backward' | 'ortho'): Tensor;
    rfftfreq(n: number, d?: number): Tensor;
    rfftn(input: Tensor, s?: number[], dim?: number | number[], norm?: 'forward' | 'backward' | 'ortho'): Tensor;
    stft(input: Tensor, n_fft: number, hop_length?: number, win_length?: number, window?: Tensor, center?: boolean, pad_mode?: 'constant' | 'reflect' | 'replicate' | 'circular', normalized?: boolean, onesided?: boolean, return_complex?: boolean): Tensor;
}

// nn.functional namespace
export interface NNFunctional {
    relu(self: Tensor): Tensor;
    sigmoid(self: Tensor, out?: Tensor): Tensor;
    tanh(self: Tensor, out?: Tensor): Tensor;
    dropout(self: Tensor, p?: number, training?: boolean): Tensor;
    elu(self: Tensor, alpha?: number): Tensor;
    gelu(self: Tensor, approximate?: 'none' | 'tanh'): Tensor;
    hardtanh(self: Tensor, minVal?: number, maxVal?: number): Tensor;
    leakyRelu(self: Tensor, negativeSlope?: number): Tensor;
    logSoftmax(self: Tensor, dim?: number, dtype?: DType): Tensor;
    logsigmoid(self: Tensor, out?: Tensor): Tensor;
    selu(self: Tensor, inplace?: boolean): Tensor;
    silu(self: Tensor): Tensor;
    softmax(self: Tensor, dim?: number, dtype?: DType): Tensor;
    softmin(self: Tensor, dim?: number, dtype?: DType): Tensor;
    linear(input: Tensor, weight: Tensor, bias?: Tensor): Tensor;
    batchNorm(self: Tensor, runningMean?: Tensor, runningVar?: Tensor, weight?: Tensor, bias?: Tensor, training?: boolean, momentum?: number, eps?: number): Tensor;
    groupNorm(self: Tensor, numGroups: number, weight?: Tensor, bias?: Tensor, eps?: number): Tensor;
    layerNorm(self: Tensor, normalizedShape: number[], weight?: Tensor, bias?: Tensor, eps?: number): Tensor;
    normalize(self: Tensor, p?: number, dim?: number, eps?: number, out?: Tensor): Tensor;
    rmsNorm(self: Tensor, normalizedShape: number[], weight?: Tensor, eps?: number): Tensor;
    adaptiveAvgPool2d(input: Tensor, outputSize: number | number[]): Tensor;
    adaptiveMaxPool2d(input: Tensor, outputSize: number | number[], returnIndices?: boolean): Tensor;
    avgPool1d(input: Tensor, kernelSize: number, stride?: number, padding?: number, ceilMode?: boolean, countIncludePad?: boolean): Tensor;
    avgPool2d(input: Tensor, kernelSize: number | number[], stride?: number | number[], padding?: number | number[], ceilMode?: boolean, countIncludePad?: boolean, divisorOverride?: number): Tensor;
    avgPool3d(input: Tensor, kernelSize: number | number[], stride?: number | number[], padding?: number | number[], ceilMode?: boolean, countIncludePad?: boolean, divisorOverride?: number): Tensor;
    conv1d(input: Tensor, weight: Tensor, bias?: Tensor, stride?: number, padding?: number | 'same' | 'valid', dilation?: number, groups?: number): Tensor;
    conv2d(input: Tensor, weight: Tensor, bias?: Tensor, stride?: number | number[], padding?: number | number[] | 'same' | 'valid', dilation?: number | number[], groups?: number): Tensor;
    conv3d(input: Tensor, weight: Tensor, bias?: Tensor, stride?: number | number[], padding?: number | number[] | 'same' | 'valid', dilation?: number | number[], groups?: number): Tensor;
    convTranspose2d(input: Tensor, weight: Tensor, bias?: Tensor, stride?: number | number[], padding?: number | number[], outputPadding?: number | number[], groups?: number, dilation?: number | number[]): Tensor;
    maxPool1d(input: Tensor, kernelSize: number, stride?: number, padding?: number, dilation?: number, ceilMode?: boolean, returnIndices?: boolean): Tensor;
    maxPool2d(input: Tensor, kernelSize: number | number[], stride?: number | number[], padding?: number | number[], dilation?: number | number[], ceilMode?: boolean, returnIndices?: boolean): Tensor;
    maxPool3d(input: Tensor, kernelSize: number | number[], stride?: number | number[], padding?: number | number[], dilation?: number | number[], ceilMode?: boolean, returnIndices?: boolean): Tensor;
    embedding(input: Tensor, weight: Tensor, paddingIdx?: number, maxNorm?: number, normType?: number, scaleGradByFreq?: boolean, sparse?: boolean): Tensor;
    scaledDotProductAttention(query: Tensor, key: Tensor, value: Tensor, attnMask?: Tensor, dropoutP?: number, isCausal?: boolean, scale?: number): Tensor;
}
