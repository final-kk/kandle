/**
 * v6 FactoryHandler
 *
 * 处理创建新 Tensor 的操作: zeros, ones, empty, randn, rand, arange, eye 等
 * 
 * 使用统一的 operators.find 访问所有 kernel
 */

import type { ITensorHandle, DType, DeviceNameEnum, IteratorKernelImpl } from '@kandle/types';
import type { OpEntry } from '@kandle/types';
import { env } from '../../env';
import { TensorIterator } from '../TensorIterator';
import type { PatternHandler, OperatorContext, DirectContext } from './types';
import { FFTHandler } from './fft';

export class FactoryHandler implements PatternHandler {
    private static instance: FactoryHandler;

    static getInstance(): FactoryHandler {
        if (!FactoryHandler.instance) {
            FactoryHandler.instance = new FactoryHandler();
        }
        return FactoryHandler.instance;
    }

    buildContext(entry: OpEntry, ctx: OperatorContext): DirectContext {
        return {
            kind: 'direct',
            inputs: ctx.tensorInputs,
            scalars: ctx.scalarArgs,
            metadata: ctx.metadata,
            outs: ctx.outs,
            kernelName: entry.dispatchKey,
        };
    }

    execute(execCtx: DirectContext): ITensorHandle {
        const { inputs, scalars, metadata, kernelName } = execCtx;
        const allParams = { ...scalars, ...metadata };
        const device = (allParams['device'] as DeviceNameEnum) ?? env.getDefaultDevice().name;
        const backend = env.getBackend(device);
        const dtype = (allParams['dtype'] ?? 'float32') as DType;

        switch (kernelName) {
            case 'zeros':
            case 'ones':
            case 'empty':
            case 'full': {
                const shape = allParams['size'] as number[];
                // 使用 TensorIterator + kernel 填充值
                const iter = TensorIterator.nullaryOp(shape, dtype, kernelName);
                if (kernelName === 'full') {
                    // 支持两种命名: fillValue (camelCase) 或 fill_value (snake_case)
                    const fillVal = (allParams['fillValue'] ?? allParams['fill_value']) as number;
                    iter.setScalarArgs({ fill_value: fillVal });
                }
                const kernel = backend.operators.find(kernelName) as IteratorKernelImpl | undefined;
                if (kernel) {
                    kernel(iter);
                }
                return iter.output().tensorHandle!;
            }

            case 'zeros_like':
            case 'ones_like':
            case 'empty_like': {
                const input = inputs[0];
                const dt = (allParams['dtype'] as DType) ?? input.dtype;
                const shape = [...input.shape];
                const opName = kernelName.replace('_like', '');
                const iter = TensorIterator.nullaryOp(shape, dt, opName);
                const kernel = backend.operators.find(opName) as IteratorKernelImpl | undefined;
                if (kernel) kernel(iter);
                return iter.output().tensorHandle!;
            }

            case 'eye': {
                // eye 使用 DirectContext 模式
                const n = allParams['n'] as number;
                const m = (allParams['m'] ?? n) as number;
                const shape = [n, m];
                const output = backend.createTensorHandle(shape, dtype);

                // 构建 DirectContext 透传给 kernel
                const eyeCtx: DirectContext = {
                    kind: 'direct',
                    inputs: [],
                    scalars: { n, m },
                    metadata: { n, m },
                    outs: [output],
                    kernelName: 'eye',
                };

                // 通过 operators.find 获取 kernel 并透传 context
                const kernel = backend.operators.find('eye');
                if (!kernel) {
                    throw new Error(`Eye kernel not available on backend`);
                }

                (kernel as (ctx: DirectContext) => void)(eyeCtx);

                return output;
            }

            case 'arange':
            case 'linspace':
            case 'rand':
            case 'randn':
            case 'randint': {
                // 根据不同操作确定 shape
                let shape: number[];
                if (kernelName === 'arange') {
                    // PyTorch parameter normalization:
                    // arange(end) → start=0, end=end
                    // arange(start, end) → start=start, end=end
                    let start = allParams['start'] as number;
                    let end = allParams['end'] as number | undefined;
                    const step = (allParams['step'] ?? 1) as number;

                    if (end === undefined || end === null) {
                        // Single argument case: interpret as end
                        end = start;
                        start = 0;
                    }

                    const numel = Math.max(0, Math.ceil((end - start) / step));
                    shape = [numel];

                    // Update allParams for kernel
                    allParams['start'] = start;
                    allParams['end'] = end;
                } else if (kernelName === 'linspace') {
                    shape = [allParams['steps'] as number];
                } else {
                    shape = allParams['size'] as number[];
                }
                const iter = TensorIterator.nullaryOp(shape, dtype, kernelName);
                // Merge metadata into scalarArgs (e.g., randint has low/high in metadata)
                const mergedScalars = { ...scalars, ...metadata } as Record<string, number | boolean | string>;
                iter.setScalarArgs(mergedScalars);
                const kernel = backend.operators.find(kernelName) as IteratorKernelImpl | undefined;
                if (!kernel) {
                    throw new Error(`Factory kernel '${kernelName}' not found`);
                }
                kernel(iter);
                return iter.output().tensorHandle!;
            }

            case 'multinomial': {
                // multinomial 使用 Direct kernel 模式
                // 类似于 eye，但需要输入 tensor
                const input = inputs[0];
                const numSamples = allParams['numSamples'] as number;
                const replacement = (allParams['replacement'] ?? false) as boolean;

                // 获取 kernel
                const refBackend = env.getBackend(input.device);
                const kernel = refBackend.operators.find('multinomial');
                if (!kernel) {
                    throw new Error(`Multinomial kernel not available on backend`);
                }

                // 调用 kernel: (input, scalars, outs?) => ITensorHandle
                const result = (kernel as any)(
                    input,
                    { numSamples, replacement },
                    undefined
                ) as ITensorHandle;

                return result;
            }

            case 'pad': {
                // N-dimensional padding
                // PyTorch F.pad: pad tensor from LAST dimension backwards
                // pad = [left, right, top, bottom, front, back, ...]
                const input = inputs[0];
                const padSizes = allParams['pad'] as number[];
                const mode = (allParams['mode'] ?? 'constant') as string;
                const value = (allParams['value'] ?? 0) as number;

                // Compute output shape and left/right pads per dimension
                const inputShape = input.shape;
                const ndim = inputShape.length;
                const outputShape = [...inputShape];
                const leftPads: number[] = new Array(ndim).fill(0);
                const rightPads: number[] = new Array(ndim).fill(0);

                for (let i = 0; i < padSizes.length; i += 2) {
                    const dimFromEnd = Math.floor(i / 2);
                    const dim = ndim - 1 - dimFromEnd;
                    if (dim >= 0) {
                        const left = padSizes[i] ?? 0;
                        const right = padSizes[i + 1] ?? 0;
                        outputShape[dim] += left + right;
                        leftPads[dim] = left;
                        rightPads[dim] = right;
                    }
                }

                const inputBackend = env.getBackend(input.device);

                if (mode === 'constant') {
                    // Create output tensor filled with pad value
                    let output: ITensorHandle;
                    if (value === 0) {
                        const zerosIter = TensorIterator.nullaryOp(outputShape, input.dtype, 'zeros');
                        const zerosKernel = inputBackend.operators.find('zeros') as IteratorKernelImpl | undefined;
                        if (zerosKernel) zerosKernel(zerosIter);
                        output = zerosIter.output().tensorHandle!;
                    } else {
                        const fullIter = TensorIterator.nullaryOp(outputShape, input.dtype, 'full');
                        fullIter.setScalarArgs({ fill_value: value });
                        const fullKernel = inputBackend.operators.find('full') as IteratorKernelImpl | undefined;
                        if (fullKernel) fullKernel(fullIter);
                        output = fullIter.output().tensorHandle!;
                    }

                    // Create a strided view of output where input data should go
                    const newOffset = leftPads.reduce((acc, left, d) => {
                        return acc + left * (output.strides[d] as number);
                    }, output.offset);

                    const targetView = inputBackend.createTensorHandle({
                        storage: output.storage,
                        shape: inputShape,
                        dtype: output.dtype,
                        strides: output.strides as number[],
                        offset: newOffset,
                    });

                    // Copy input to the target view
                    const copyIter = TensorIterator.unaryOp(input, 'contiguous', targetView);
                    const copyKernel = inputBackend.operators.find('contiguous') as IteratorKernelImpl | undefined;
                    if (copyKernel) {
                        copyKernel(copyIter);
                    } else {
                        throw new Error('pad: contiguous kernel not found');
                    }

                    return output;

                } else if (mode === 'replicate' || mode === 'reflect') {
                    // For replicate/reflect, we need to compute padded indices on CPU
                    // then use a copy/gather approach

                    // Create output tensor
                    const output = inputBackend.createTensorHandle(outputShape, input.dtype);

                    // Use the pad kernel directly if available (Direct kernel approach)
                    const padKernel = inputBackend.operators.find('pad_' + mode);
                    if (padKernel) {
                        // Direct kernel exists
                        const ctx: DirectContext = {
                            kind: 'direct',
                            inputs: [input],
                            scalars: { leftPads, rightPads, mode },
                            metadata: {},
                            outs: [output],
                            kernelName: 'pad_' + mode,
                        };
                        (padKernel as (ctx: DirectContext) => void)(ctx);
                        return output;
                    }

                    // Fallback: CPU-based index computation + element-wise copy via contiguous
                    // For 1D padding only (audio use case), implement a simpler version

                    // Check if only last dimension is padded (common for audio)
                    const paddedDims = leftPads.map((l, i) => l > 0 || rightPads[i] > 0 ? i : -1).filter(i => i >= 0);

                    if (paddedDims.length === 1 && paddedDims[0] === ndim - 1) {
                        // 1D padding on last dimension - can use slice + cat approach
                        const dim = paddedDims[0];
                        const leftPad = leftPads[dim];
                        const rightPad = rightPads[dim];
                        const inputLen = inputShape[dim];

                        const parts: ITensorHandle[] = [];

                        // Left padding: replicate first element / reflect from start
                        if (leftPad > 0) {
                            if (mode === 'replicate') {
                                // Take first element and repeat it leftPad times
                                // Use expand with stride=0
                                const firstSliceShape = [...inputShape];
                                firstSliceShape[dim] = 1;

                                // Create view of first element
                                const firstView = inputBackend.createTensorHandle({
                                    storage: input.storage,
                                    shape: firstSliceShape,
                                    dtype: input.dtype,
                                    strides: input.strides as number[],
                                    offset: input.offset,
                                });

                                // Expand to leftPad size with stride=0 on that dimension
                                const expandedShape = [...inputShape];
                                expandedShape[dim] = leftPad;
                                const expandedStrides = [...input.strides] as number[];
                                expandedStrides[dim] = 0; // Broadcast

                                const leftExpanded = inputBackend.createTensorHandle({
                                    storage: input.storage,
                                    shape: expandedShape,
                                    dtype: input.dtype,
                                    strides: expandedStrides,
                                    offset: input.offset,
                                });

                                // Make contiguous
                                const leftPart = inputBackend.createTensorHandle(expandedShape, input.dtype);
                                const leftCopyIter = TensorIterator.unaryOp(leftExpanded, 'contiguous', leftPart);
                                const copyKernel = inputBackend.operators.find('contiguous') as IteratorKernelImpl | undefined;
                                if (copyKernel) copyKernel(leftCopyIter);
                                parts.push(leftPart);

                            } else { // reflect
                                // Reflect: indices 1, 2, ..., leftPad (reversed)
                                // For leftPad elements, we take input[leftPad], input[leftPad-1], ..., input[1]
                                // This requires flip + slice

                                // Simpler: build each slice individually (less efficient but correct)
                                // For now, create reflected data on CPU and upload
                                // This is a reasonable fallback for small padding sizes

                                const reflectShape = [...inputShape];
                                reflectShape[dim] = leftPad;
                                const leftPart = inputBackend.createTensorHandle(reflectShape, input.dtype);

                                // Use flip on input slice [1:leftPad+1]
                                // For simplicity, take slice and then reverse via strided view
                                const sliceEnd = Math.min(leftPad + 1, inputLen);
                                const sliceStart = 1;
                                const sliceLen = sliceEnd - sliceStart;

                                if (sliceLen > 0) {
                                    // Create reversed view
                                    const sliceOffset = input.offset + sliceStart * (input.strides[dim] as number);
                                    const reversedStrides = [...input.strides] as number[];
                                    reversedStrides[dim] = -(input.strides[dim] as number);

                                    const sliceShape = [...inputShape];
                                    sliceShape[dim] = Math.min(leftPad, sliceLen);

                                    // Offset should point to the last element of the slice
                                    const reversedOffset = sliceOffset + (sliceLen - 1) * (input.strides[dim] as number);

                                    const reversedView = inputBackend.createTensorHandle({
                                        storage: input.storage,
                                        shape: sliceShape,
                                        dtype: input.dtype,
                                        strides: reversedStrides,
                                        offset: reversedOffset,
                                    });

                                    // Copy to leftPart
                                    const reflectIter = TensorIterator.unaryOp(reversedView, 'contiguous', leftPart);
                                    const copyKernel = inputBackend.operators.find('contiguous') as IteratorKernelImpl | undefined;
                                    if (copyKernel) copyKernel(reflectIter);
                                }

                                parts.push(leftPart);
                            }
                        }

                        // Middle: the original input
                        parts.push(input);

                        // Right padding: replicate last element / reflect from end
                        if (rightPad > 0) {
                            if (mode === 'replicate') {
                                // Take last element and repeat it rightPad times
                                const lastOffset = input.offset + (inputLen - 1) * (input.strides[dim] as number);

                                const expandedShape = [...inputShape];
                                expandedShape[dim] = rightPad;
                                const expandedStrides = [...input.strides] as number[];
                                expandedStrides[dim] = 0;

                                const rightExpanded = inputBackend.createTensorHandle({
                                    storage: input.storage,
                                    shape: expandedShape,
                                    dtype: input.dtype,
                                    strides: expandedStrides,
                                    offset: lastOffset,
                                });

                                const rightPart = inputBackend.createTensorHandle(expandedShape, input.dtype);
                                const rightCopyIter = TensorIterator.unaryOp(rightExpanded, 'contiguous', rightPart);
                                const copyKernel = inputBackend.operators.find('contiguous') as IteratorKernelImpl | undefined;
                                if (copyKernel) copyKernel(rightCopyIter);
                                parts.push(rightPart);

                            } else { // reflect
                                // Similar to left, but from the end
                                const reflectShape = [...inputShape];
                                reflectShape[dim] = rightPad;
                                const rightPart = inputBackend.createTensorHandle(reflectShape, input.dtype);

                                const sliceStart = inputLen - 2;
                                const sliceEnd = Math.max(inputLen - rightPad - 2, -1);
                                const sliceLen = sliceStart - sliceEnd;

                                if (sliceLen > 0) {
                                    const sliceOffset = input.offset + sliceStart * (input.strides[dim] as number);
                                    const reversedStrides = [...input.strides] as number[];
                                    reversedStrides[dim] = -(input.strides[dim] as number);

                                    const sliceShape = [...inputShape];
                                    sliceShape[dim] = Math.min(rightPad, sliceLen);

                                    const reversedView = inputBackend.createTensorHandle({
                                        storage: input.storage,
                                        shape: sliceShape,
                                        dtype: input.dtype,
                                        strides: reversedStrides,
                                        offset: sliceOffset,
                                    });

                                    const reflectIter = TensorIterator.unaryOp(reversedView, 'contiguous', rightPart);
                                    const copyKernel = inputBackend.operators.find('contiguous') as IteratorKernelImpl | undefined;
                                    if (copyKernel) copyKernel(reflectIter);
                                }

                                parts.push(rightPart);
                            }
                        }

                        // Cat all parts along the padded dimension
                        // Need to use cat operation
                        if (parts.length === 1) {
                            return parts[0];
                        }

                        // Use cat via shape handler (import would cause circular dep)
                        // Instead, manually concatenate by computing offsets
                        let currentOffset = 0;
                        for (const part of parts) {
                            const partLen = part.shape[dim];
                            const targetOffset = output.offset + currentOffset * (output.strides[dim] as number);

                            const targetView = inputBackend.createTensorHandle({
                                storage: output.storage,
                                shape: [...part.shape],
                                dtype: output.dtype,
                                strides: output.strides as number[],
                                offset: targetOffset,
                            });

                            const catCopyIter = TensorIterator.unaryOp(part, 'contiguous', targetView);
                            const copyKernel = inputBackend.operators.find('contiguous') as IteratorKernelImpl | undefined;
                            if (copyKernel) copyKernel(catCopyIter);

                            currentOffset += partLen;
                        }

                        return output;

                    } else {
                        // Multi-dimensional replicate/reflect - not yet implemented
                        throw new Error(`pad: mode '${mode}' for multi-dimensional padding not yet implemented`);
                    }

                } else {
                    throw new Error(`pad: mode '${mode}' not supported. Supported modes: constant, replicate, reflect`);
                }
            }

            case 'fft':
            case 'ifft':
            case 'rfft':
            case 'irfft': {
                // 委托给 FFTHandler 处理 FFT 相关操作
                return FFTHandler.getInstance().execute(execCtx);
            }

            default:
                throw new Error(`Unknown factory operation: ${kernelName}`);
        }
    }

    static dispatch(entry: OpEntry, ctx: OperatorContext): ITensorHandle {
        const handler = FactoryHandler.getInstance();
        const execCtx = handler.buildContext(entry, ctx);
        return handler.execute(execCtx);
    }
}

export const dispatchFactory = FactoryHandler.dispatch;
