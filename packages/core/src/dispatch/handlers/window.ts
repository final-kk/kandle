/**
 * Window Handler - Conv/Pool 操作处理器
 * 
 * 处理 ComputePattern = 'Window' 的操作，包括：
 * - 卷积: conv1d, conv2d, conv3d, convTranspose2d
 * - 池化: maxPool1d/2d/3d, avgPool1d/2d/3d
 * - 自适应池化: adaptiveAvgPool2d, adaptiveMaxPool2d
 * 
 * @module dispatch/handlers/window
 */

import type { ITensorHandle, DType, MemoryFormat } from '@kandle/types';
import type { OpEntry } from '@kandle/types';
import type { OperatorContext, PatternHandler, ExecutionContext, DirectContext } from './types';
import { env } from '../../env';
import { inferMemoryFormat, isChannelsLast as isChannelsLastFormat } from '@kandle/utils';

// ============================================================================
// Conv Dispatch Result - 传递给 kernel 的配置
// ============================================================================

/**
 * Conv 操作类型
 */
export type ConvVariant =
    | 'conv1d' | 'conv2d' | 'conv3d'
    | 'conv_transpose2d' | 'conv_transpose3d';

/**
 * Pool 操作类型
 */
export type PoolVariant =
    | 'max_pool1d' | 'max_pool2d' | 'max_pool3d'
    | 'avg_pool1d' | 'avg_pool2d' | 'avg_pool3d'
    | 'adaptive_avg_pool2d' | 'adaptive_max_pool2d';

/**
 * 卷积算法选择
 */
export type ConvAlgorithm = 'im2col' | 'direct' | 'winograd' | 'fft';

/**
 * Conv/Pool 操作的完整配置
 */
export interface ConvDispatchResult {
    /** 操作类型 */
    variant: ConvVariant | PoolVariant;

    /** 输出张量 */
    output: ITensorHandle;

    // === 输入信息 ===
    batchSize: number;
    inChannels: number;
    outChannels: number;
    /** 输入空间维度 [H, W] 或 [L] 或 [D, H, W] */
    inputSpatial: number[];
    /** 输出空间维度 */
    outputSpatial: number[];

    // === 卷积/池化参数 ===
    kernelSize: number[];
    stride: number[];
    padding: number[];
    dilation: number[];
    groups: number;

    // === 可选输入 ===
    bias?: ITensorHandle;

    // === Pooling 特有字段 ===
    /** 是否返回最大值索引 (仅 max_pool) */
    returnIndices?: boolean;
    /** 索引输出张量 (dtype: int64) */
    indicesOutput?: ITensorHandle;

    // === 算法选择 (仅 Conv) ===
    algorithm?: ConvAlgorithm;

    /** 计算类型 */
    computeDtype: DType;

    // === MemoryFormat ===
    inputFormat: MemoryFormat;
    outputFormat: MemoryFormat;
    isChannelsLast: boolean;
}

// ============================================================================
// Window Handler
// ============================================================================

export class WindowHandler implements PatternHandler {
    private static instance: WindowHandler;

    static getInstance(): WindowHandler {
        if (!WindowHandler.instance) {
            WindowHandler.instance = new WindowHandler();
        }
        return WindowHandler.instance;
    }

    buildContext(entry: OpEntry, ctx: OperatorContext): ExecutionContext {
        const { opName, tensorInputs, scalarArgs, metadata, outs } = ctx;

        // 根据 dispatchKey 路由到对应的处理函数
        switch (entry.dispatchKey) {
            case 'conv1d':
            case 'conv2d':
            case 'conv3d':
            case 'conv_transpose2d':
                return buildConvContext(entry, ctx);

            case 'max_pool1d':
            case 'max_pool2d':
            case 'max_pool3d':
            case 'avg_pool1d':
            case 'avg_pool2d':
            case 'avg_pool3d':
            case 'adaptive_avg_pool2d':
            case 'adaptive_max_pool2d':
                return buildPoolContext(entry, ctx);

            default:
                throw new Error(`WindowHandler: Unknown dispatchKey "${entry.dispatchKey}"`);
        }
    }

    execute(execCtx: ExecutionContext): ITensorHandle | ITensorHandle[] {
        if (execCtx.kind !== 'direct') {
            throw new Error('WindowHandler expects DirectContext');
        }

        const { kernelName, inputs, scalars, outs } = execCtx as DirectContext;
        const backend = env.getDefaultDevice();
        const kernel = backend.operators.find(kernelName);

        if (!kernel) {
            throw new Error(`Kernel "${kernelName}" not registered in backend "${backend.name}"`);
        }

        // 调用 kernel，传递 ConvDispatchResult 配置和输入
        const config = scalars['__config'] as ConvDispatchResult;
        // 使用 Function.prototype.call 避免类型问题
        const result = (kernel as Function)(config, ...inputs);

        // 确保返回类型正确
        if (result === undefined || result === null) {
            throw new Error(`Kernel "${kernelName}" returned undefined`);
        }
        return result as ITensorHandle | ITensorHandle[];
    }

    static dispatch(entry: OpEntry, ctx: OperatorContext): ITensorHandle | ITensorHandle[] {
        const handler = WindowHandler.getInstance();
        const execCtx = handler.buildContext(entry, ctx);
        return handler.execute(execCtx);
    }
}

// ============================================================================
// Conv Context Builder
// ============================================================================

function buildConvContext(entry: OpEntry, ctx: OperatorContext): DirectContext {
    const { tensorInputs, scalarArgs, metadata, outs } = ctx;
    const args = { ...scalarArgs, ...metadata };
    const [input, weight] = tensorInputs;
    const bias = tensorInputs.length > 2 ? tensorInputs[2] : undefined;

    // 解析卷积参数（确保类型正确）
    const ndim = getSpatialDims(entry.dispatchKey);
    const strideRaw = args['stride'] ?? 1;
    const stride = normalizeIntOrList(strideRaw as number | number[], ndim);

    const dilationRaw = args['dilation'] ?? 1;
    const dilation = normalizeIntOrList(dilationRaw as number | number[], ndim);

    const paddingRaw = args['padding'] ?? 0;
    const padding = normalizePadding(paddingRaw as number | number[] | string, ndim, input.shape, weight.shape, stride, dilation);

    const groups = (args['groups'] ?? 1) as number;

    // 验证参数
    validateConvParams(input, weight, groups, ndim, entry.dispatchKey);

    // 计算输出形状
    const outputShape = computeConvOutputShape(
        input.shape,
        weight.shape,
        stride,
        padding,
        dilation,
        groups,
        entry.dispatchKey
    );

    // 检测内存格式
    // @ts-ignore - ITensorHandle definition update might not be picked up yet
    const inputFormat = (input as any).memoryFormat;
    const isChannelsLast = isChannelsLastFormat(inputFormat);
    const outputFormat = inputFormat; // 保持输入格式

    // 创建输出张量
    const backend = env.getDefaultDevice();
    const output = outs?.[0] ?? backend.createTensorHandle({
        shape: outputShape,
        dtype: input.dtype,
        memoryFormat: outputFormat,
    });

    // 选择算法
    const algorithm = selectConvAlgorithm(
        weight.shape.slice(-ndim), // kernel size
        stride, dilation, groups
    );

    // 构建 ConvDispatchResult
    const config: ConvDispatchResult = {
        variant: entry.dispatchKey as ConvVariant,
        output,
        batchSize: input.shape[0],
        inChannels: input.shape[1],
        outChannels: weight.shape[0],
        inputSpatial: input.shape.slice(2) as number[],
        outputSpatial: outputShape.slice(2) as number[],
        kernelSize: weight.shape.slice(-ndim) as number[],
        stride,
        padding,
        dilation,
        groups,
        bias,
        algorithm,
        computeDtype: input.dtype,
        inputFormat,
        outputFormat,
        isChannelsLast,
    };

    return {
        kind: 'direct',
        inputs: [input, weight, ...(bias ? [bias] : [])],
        scalars: { '__config': config },
        outs: [output],
        kernelName: entry.dispatchKey,
    };
}

// ============================================================================
// Pool Context Builder
// ============================================================================

function buildPoolContext(entry: OpEntry, ctx: OperatorContext): DirectContext {
    const { tensorInputs, scalarArgs, metadata, outs } = ctx;
    const args = { ...scalarArgs, ...metadata };
    const [input] = tensorInputs;

    const isAdaptive = entry.dispatchKey.includes('adaptive');
    const ndim = getSpatialDims(entry.dispatchKey);

    let kernelSize: number[];
    let stride: number[];
    let padding: number[];
    let dilation: number[];
    let outputShape: number[];

    if (isAdaptive) {
        // 自适应池化：根据目标输出尺寸计算 kernel/stride
        const outputSizeRaw = args['outputSize'] as number | number[];
        const outputSize = normalizeIntOrList(outputSizeRaw, ndim);
        const inputSpatial = input.shape.slice(2) as number[];

        kernelSize = inputSpatial.map((s, i) => Math.ceil(s / outputSize[i]));
        stride = inputSpatial.map((s, i) => Math.floor(s / outputSize[i]));
        padding = new Array(ndim).fill(0);
        dilation = new Array(ndim).fill(1);
        outputShape = [...input.shape.slice(0, 2), ...outputSize];
    } else {
        const kernelSizeRaw = args['kernelSize'] as number | number[] | undefined;
        if (kernelSizeRaw === undefined) {
            throw new Error(`Pooling op ${entry.name} requires 'kernelSize'`);
        }
        kernelSize = normalizeIntOrList(kernelSizeRaw, ndim);

        const strideRaw = args['stride'];
        // Stride defaults to kernelSize if not provided
        stride = strideRaw !== undefined
            ? normalizeIntOrList(strideRaw as number | number[], ndim)
            : [...kernelSize]; // Copy to avoid reference issues

        const paddingRaw = args['padding'] ?? 0;
        padding = normalizeIntOrList(paddingRaw as number | number[], ndim);

        const dilationRaw = args['dilation'] ?? 1;
        dilation = normalizeIntOrList(dilationRaw as number | number[], ndim);

        const ceilMode = (args['ceilMode'] ?? false) as boolean;
        outputShape = computePoolOutputShape(
            input.shape, kernelSize, stride, padding, dilation, ceilMode
        );
    }

    // 检测内存格式
    const inputFormat = input.memoryFormat;
    const isChannelsLast = isChannelsLastFormat(inputFormat);
    const outputFormat = inputFormat;

    // 解析 returnIndices (仅 max_pool)
    const isMaxPool = entry.dispatchKey.includes('max');
    const returnIndices = isMaxPool && (args['returnIndices'] ?? false) as boolean;

    // 创建输出张量
    const backend = env.getDefaultDevice();
    const output = outs?.[0] ?? backend.createTensorHandle({
        shape: outputShape,
        dtype: input.dtype,
        memoryFormat: outputFormat,
    });

    // 如果需要返回索引，创建 indices 输出张量 (int64)
    let indicesOutput: ITensorHandle | undefined;
    if (returnIndices) {
        indicesOutput = outs?.[1] ?? backend.createTensorHandle({
            shape: outputShape,
            dtype: 'int64',
            memoryFormat: outputFormat,
        });
    }

    const config: ConvDispatchResult = {
        variant: entry.dispatchKey as PoolVariant,
        output,
        batchSize: input.shape[0],
        inChannels: input.shape[1],
        outChannels: input.shape[1], // 池化不改变通道数
        inputSpatial: input.shape.slice(2) as number[],
        outputSpatial: outputShape.slice(2) as number[],
        kernelSize,
        stride,
        padding,
        dilation,
        groups: 1,
        returnIndices,
        indicesOutput,
        computeDtype: input.dtype,
        inputFormat,
        outputFormat,
        isChannelsLast,
    };

    // 准备输出列表
    const outsArray = returnIndices && indicesOutput
        ? [output, indicesOutput]
        : [output];

    return {
        kind: 'direct',
        inputs: [input],
        scalars: { '__config': config },
        outs: outsArray,
        kernelName: entry.dispatchKey,
    };
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * 获取空间维度数
 */
function getSpatialDims(dispatchKey: string): number {
    if (dispatchKey.includes('1d')) return 1;
    if (dispatchKey.includes('2d')) return 2;
    if (dispatchKey.includes('3d')) return 3;
    throw new Error(`Cannot determine spatial dims for: ${dispatchKey}`);
}

/**
 * 将 int 或 int[] 规范化为固定长度数组
 */
function normalizeIntOrList(value: number | number[], length: number): number[] {
    if (typeof value === 'number') {
        return new Array(length).fill(value);
    }
    if (value.length === 1) {
        return new Array(length).fill(value[0]);
    }
    if (value.length !== length) {
        throw new Error(`Expected ${length} values, got ${value.length}`);
    }
    return value;
}

/**
 * 解析 padding 参数 ('same', 'valid', 或数值)
 */
function normalizePadding(
    padding: number | number[] | string,
    ndim: number,
    inputShape: readonly number[],
    weightShape: readonly number[],
    stride: number[],
    dilation: number | number[]
): number[] {
    if (typeof padding === 'string') {
        if (padding === 'valid') {
            return new Array(ndim).fill(0);
        }
        if (padding === 'same') {
            // 计算 'same' padding
            const dilationArr = normalizeIntOrList(dilation, ndim);
            const kernelSize = weightShape.slice(-ndim);
            return kernelSize.map((k, i) => {
                const effectiveK = (k - 1) * dilationArr[i] + 1;
                return Math.floor((effectiveK - 1) / 2);
            });
        }
        throw new Error(`Unknown padding mode: ${padding}`);
    }
    return normalizeIntOrList(padding, ndim);
}

/**
 * 验证卷积参数
 */
function validateConvParams(
    input: ITensorHandle,
    weight: ITensorHandle,
    groups: number,
    ndim: number,
    dispatchKey: string
): void {
    const inChannels = input.shape[1];
    const outChannels = weight.shape[0];
    const weightInChannels = weight.shape[1] * groups;

    // input channels 必须能被 groups 整除
    if (inChannels % groups !== 0) {
        throw new Error(
            `${dispatchKey}: input channels (${inChannels}) must be divisible by groups (${groups})`
        );
    }

    // weight 的 in_channels/groups 必须匹配
    if (weightInChannels !== inChannels) {
        throw new Error(
            `${dispatchKey}: weight in_channels (${weight.shape[1]} × groups=${groups}) ` +
            `must match input channels (${inChannels})`
        );
    }

    // output channels 必须能被 groups 整除
    if (outChannels % groups !== 0) {
        throw new Error(
            `${dispatchKey}: out_channels (${outChannels}) must be divisible by groups (${groups})`
        );
    }

    // 张量维度检查
    const expectedInputDim = ndim + 2; // N, C, spatial...
    if (input.shape.length !== expectedInputDim) {
        throw new Error(
            `${dispatchKey}: input must be ${expectedInputDim}D, got ${input.shape.length}D`
        );
    }
}

/**
 * 计算卷积输出形状
 */
function computeConvOutputShape(
    inputShape: readonly number[],
    weightShape: readonly number[],
    stride: number[],
    padding: number[],
    dilation: number[],
    groups: number,
    dispatchKey: string
): number[] {
    const [N, C_in, ...inputSpatial] = inputShape;
    const [C_out, _, ...kernelSize] = weightShape;

    const outputSpatial = inputSpatial.map((size, i) => {
        const effectiveK = (kernelSize[i] - 1) * dilation[i] + 1;
        return Math.floor((size + 2 * padding[i] - effectiveK) / stride[i]) + 1;
    });

    return [N, C_out, ...outputSpatial];
}

/**
 * 计算池化输出形状
 */
function computePoolOutputShape(
    inputShape: readonly number[],
    kernelSize: number[],
    stride: number[],
    padding: number[],
    dilation: number[],
    ceilMode: boolean
): number[] {
    const [N, C, ...inputSpatial] = inputShape;

    const outputSpatial = inputSpatial.map((size, i) => {
        const effectiveK = (kernelSize[i] - 1) * dilation[i] + 1;
        const num = size + 2 * padding[i] - effectiveK;
        const s = stride[i];
        return ceilMode
            ? Math.ceil(num / s) + 1
            : Math.floor(num / s) + 1;
    });

    return [N, C, ...outputSpatial];
}

/**
 * 选择最优卷积算法
 */
function selectConvAlgorithm(
    kernelSize: number[],
    stride: number[],
    dilation: number[],
    groups: number
): ConvAlgorithm {

    // Winograd: 仅 3x3 kernel, stride=1, dilation=1, groups=1
    if (kernelSize.length === 2 &&
        kernelSize[0] === 3 && kernelSize[1] === 3 &&
        stride[0] === 1 && stride[1] === 1 &&
        dilation[0] === 1 && dilation[1] === 1 &&
        groups === 1) {
        return 'winograd';
    }

    // Direct: 1x1 kernel (等价于逐点乘法)
    if (kernelSize.every(k => k === 1)) {
        return 'direct';
    }

    // 默认: Im2Col + GEMM
    return 'im2col';
}

// ============================================================================
// Dispatch Function
// ============================================================================

/**
 * 分发 Window 操作 (Conv/Pool)
 */
export function dispatchWindow(entry: OpEntry, ctx: OperatorContext): ITensorHandle | ITensorHandle[] {
    return WindowHandler.dispatch(entry, ctx);
}
