/**
 * v6 ShapeHandler
 *
 * 处理形状操作: 
 * - View 操作 (零拷贝): reshape, permute, transpose, squeeze, unsqueeze, expand, select, slice, diagonal
 * - Cat 操作 (需要拷贝): cat, stack
 * 
 * 对标 PyTorch aten/src/ATen/native/TensorShape.cpp
 */

import type { ITensorHandle, DType, Shape } from '@kandle/types';
import type { OpEntry } from '@kandle/types';
import type { IteratorKernelImpl } from '@kandle/types';
import { parseSliceString, computeStrides, computeSliceParams } from '@kandle/utils';
import { env } from '../../env';
import { TensorIterator } from '../TensorIterator';
import type {
    PatternHandler,
    OperatorContext,
    MetadataContext,
    DirectContext,
    ExecutionContext,
} from './types';
import { dispatchCopy } from './copy';

// ============================================================================
// ShapeHandler - 统一的形状操作处理器
// ============================================================================

/**
 * ShapeHandler - 处理所有形状相关操作
 * 
 * 包含两类操作:
 * 1. View 操作 (MetadataContext): 零拷贝，只修改 shape/strides 元数据
 * 2. Cat 操作 (DirectContext): 需要数据拷贝，使用 copy kernel
 */
export class ShapeHandler implements PatternHandler {
    private static instance: ShapeHandler;

    static getInstance(): ShapeHandler {
        if (!ShapeHandler.instance) {
            ShapeHandler.instance = new ShapeHandler();
        }
        return ShapeHandler.instance;
    }

    // View 操作列表 (零拷贝)
    private static readonly VIEW_OPS = new Set([
        'reshape', 'view', 'permute', 'transpose', 'unsqueeze', 'squeeze',
        'flatten', 'expand', 'select', 'slice', 'diagonal', 'asStrided'
    ]);

    // Cat 操作列表 (需要拷贝, 使用 copy kernel)
    private static readonly CAT_OPS = new Set(['cat', 'stack']);

    // Data Copy 操作列表 (需要拷贝, 使用专用 kernel)
    // 注意: 这里使用 dispatchKey (snake_case), 不是 opName (camelCase)
    private static readonly DATA_COPY_KERNEL_OPS = new Set(['repeat_interleave', 'diff', 'flip', 'fliplr', 'flipud']);

    buildContext(entry: OpEntry, ctx: OperatorContext): ExecutionContext {
        const opName = entry.name;
        const dispatchKey = entry.dispatchKey;

        if (ShapeHandler.VIEW_OPS.has(opName)) {
            return {
                kind: 'metadata',
                input: ctx.tensorInputs[0],
                params: { ...ctx.scalarArgs, ...ctx.metadata },
                opName: entry.name,
            } as MetadataContext;
        } else if (ShapeHandler.CAT_OPS.has(opName)) {
            return {
                kind: 'direct',
                inputs: ctx.tensorInputs,
                scalars: ctx.scalarArgs,
                metadata: ctx.metadata,
                outs: ctx.outs,
                kernelName: entry.dispatchKey,
            } as DirectContext;
        } else if (ShapeHandler.DATA_COPY_KERNEL_OPS.has(dispatchKey)) {
            // 专用 kernel 操作 (repeat_interleave 等)
            return {
                kind: 'direct',
                inputs: ctx.tensorInputs,
                scalars: ctx.scalarArgs,
                metadata: ctx.metadata,
                outs: ctx.outs,
                kernelName: entry.dispatchKey,
            } as DirectContext;
        } else {
            throw new Error(`Unknown shape operation: ${opName}`);
        }
    }

    execute(execCtx: ExecutionContext): ITensorHandle {
        if (execCtx.kind === 'metadata') {
            return this.executeView(execCtx);
        } else if (execCtx.kind === 'direct') {
            const directCtx = execCtx as DirectContext;
            // 区分 cat/stack 和专用 kernel 操作
            if (ShapeHandler.DATA_COPY_KERNEL_OPS.has(directCtx.kernelName)) {
                return this.executeKernelOp(directCtx);
            } else {
                return this.executeCat(directCtx);
            }
        }
        throw new Error(`Unsupported execution context kind: ${(execCtx as any).kind}`);
    }

    /**
     * 执行专用 kernel 操作 (repeatInterleave, flip 等)
     */
    private executeKernelOp(execCtx: DirectContext): ITensorHandle {
        const { inputs, scalars, metadata, outs, kernelName } = execCtx;

        if (inputs.length === 0) {
            throw new Error(`Shape kernel operation '${kernelName}' requires at least one input tensor`);
        }

        const backend = env.getBackend(inputs[0].device);

        // Handle fliplr/flipud: convert to flip kernel call
        let effectiveKernelName = kernelName;
        const mergedScalars: Record<string, unknown> = { ...scalars };

        if (kernelName === 'fliplr') {
            // fliplr: flip along dim=1, requires at least 2D
            if (inputs[0].shape.length < 2) {
                throw new Error(`fliplr: input must be at least 2D, got ${inputs[0].shape.length}D`);
            }
            effectiveKernelName = 'flip';
            mergedScalars['dims'] = [1];
        } else if (kernelName === 'flipud') {
            // flipud: flip along dim=0, requires at least 1D
            if (inputs[0].shape.length < 1) {
                throw new Error(`flipud: input must be at least 1D`);
            }
            effectiveKernelName = 'flip';
            mergedScalars['dims'] = [0];
        }

        const kernel = backend.operators.find(effectiveKernelName);

        if (!kernel) {
            throw new Error(
                `Shape kernel '${effectiveKernelName}' not implemented for backend '${inputs[0].device}'.`
            );
        }

        // 合并 metadata
        if (metadata) {
            Object.assign(mergedScalars, metadata);
        }

        // 调用 kernel: (inputs, scalars, outs?) => ITensorHandle
        const result = (kernel as any)(
            inputs,
            mergedScalars,
            outs
        ) as ITensorHandle;

        return result;
    }

    /**
     * 一键执行: build + execute
     */
    static dispatch(entry: OpEntry, ctx: OperatorContext): ITensorHandle {
        const handler = ShapeHandler.getInstance();
        const execCtx = handler.buildContext(entry, ctx);
        return handler.execute(execCtx);
    }

    // ========================================================================
    // View Operations (Zero-Copy)
    // ========================================================================

    private executeView(execCtx: MetadataContext): ITensorHandle {
        const { input, params, opName } = execCtx;
        const backend = env.getBackend(input.device);

        switch (opName) {
            case 'reshape':
            case 'view':
                return this.reshape(input, params['shape'] as number[], backend);

            case 'permute':
                return this.permute(input, params['dims'] as number[], backend);

            case 'transpose':
                return this.transpose(
                    input,
                    params['dim0'] as number,
                    params['dim1'] as number,
                    backend
                );

            case 'unsqueeze':
                return this.unsqueeze(input, params['dim'] as number, backend);

            case 'squeeze':
                return this.squeeze(input, params['dim'] as number | undefined, backend);

            case 'flatten':
                return this.flatten(
                    input,
                    params['startDim'] as number ?? 0,
                    params['endDim'] as number ?? -1,
                    backend
                );

            case 'expand':
                return this.expand(input, params['size'] as number[], backend);

            case 'select':
                return this.select(
                    input,
                    params['dim'] as number,
                    params['index'] as number,
                    backend
                );

            case 'slice': {
                const slices = params['slices'] as string | number;

                // 整数索引: 转换为 select(0, index)，降维
                if (typeof slices === 'number') {
                    return this.select(input, 0, slices, backend);
                }

                // 字符串: 解析 Python 风格切片字符串
                const parsed = parseSliceString(slices, input.shape);
                const starts = parsed.map(p => p.start);
                const ends = parsed.map(p => p.end);
                const steps = parsed.map(p => p.step);
                return this.sliceByArrays(input, starts, ends, steps, backend);
            }

            case 'diagonal':
                return this.diagonal(
                    input,
                    params['offset'] as number,
                    params['dim1'] as number,
                    params['dim2'] as number,
                    backend
                );

            case 'asStrided':
                return this.asStrided(
                    input,
                    params['size'] as number[],
                    params['stride'] as number[],
                    params['storageOffset'] as number | undefined,
                    backend
                );

            default:
                throw new Error(`Unknown view operation: ${opName}`);
        }
    }

    private reshape(
        input: ITensorHandle,
        shape: number[],
        backend: ReturnType<typeof env.getBackend>
    ): ITensorHandle {
        const newShape = this.normalizeShape(shape, input.numel);

        if (this.canView(input, newShape)) {
            // 零拷贝
            return backend.createTensorHandle({
                storage: input.storage,
                shape: newShape,
                dtype: input.dtype,
                strides: computeStrides(newShape),
                offset: input.offset,
            });
        } else {
            // 需要先 contiguous - 使用 CopyHandler
            const contiguousTensor = dispatchCopy(
                { dispatchKey: 'contiguous' } as OpEntry,
                {
                    opName: 'contiguous',
                    tensorInputs: [input],
                    scalarArgs: {},
                    metadata: {},
                    outs: undefined,
                }
            );
            return backend.createTensorHandle({
                storage: contiguousTensor.storage,
                shape: newShape,
                dtype: input.dtype,
                strides: computeStrides(newShape),
                offset: 0,
            });
        }
    }

    private permute(
        input: ITensorHandle,
        dims: number[],
        backend: ReturnType<typeof env.getBackend>
    ): ITensorHandle {
        const ndim = input.shape.length;
        const normalizedDims = dims.map(d => (d < 0 ? ndim + d : d));

        // 验证 dims
        if (normalizedDims.length !== ndim) {
            throw new Error(`permute: dims length ${dims.length} != ndim ${ndim}`);
        }

        const newShape = normalizedDims.map(d => input.shape[d]);
        const newStrides = normalizedDims.map(d => input.strides[d]);

        return backend.createTensorHandle({
            storage: input.storage,
            shape: newShape,
            dtype: input.dtype,
            strides: newStrides,
            offset: input.offset,
        });
    }

    private transpose(
        input: ITensorHandle,
        dim0: number,
        dim1: number,
        backend: ReturnType<typeof env.getBackend>
    ): ITensorHandle {
        const ndim = input.shape.length;
        const d0 = dim0 < 0 ? ndim + dim0 : dim0;
        const d1 = dim1 < 0 ? ndim + dim1 : dim1;

        const newShape = [...input.shape];
        const newStrides = [...input.strides];

        [newShape[d0], newShape[d1]] = [newShape[d1], newShape[d0]];
        [newStrides[d0], newStrides[d1]] = [newStrides[d1], newStrides[d0]];

        return backend.createTensorHandle({
            storage: input.storage,
            shape: newShape,
            dtype: input.dtype,
            strides: newStrides,
            offset: input.offset,
        });
    }

    private unsqueeze(
        input: ITensorHandle,
        dim: number,
        backend: ReturnType<typeof env.getBackend>
    ): ITensorHandle {
        const ndim = input.shape.length;
        const d = dim < 0 ? ndim + 1 + dim : dim;

        const newShape = [...input.shape];
        const newStrides = [...input.strides];

        newShape.splice(d, 0, 1);
        // stride for size-1 dim doesn't matter, use 1
        newStrides.splice(d, 0, 1);

        return backend.createTensorHandle({
            storage: input.storage,
            shape: newShape,
            dtype: input.dtype,
            strides: newStrides,
            offset: input.offset,
        });
    }

    private squeeze(
        input: ITensorHandle,
        dim: number | undefined,
        backend: ReturnType<typeof env.getBackend>
    ): ITensorHandle {
        const shape = [...input.shape];
        const strides = [...input.strides];

        if (dim === undefined) {
            // squeeze all size-1 dims
            const newShape: number[] = [];
            const newStrides: number[] = [];
            for (let i = 0; i < shape.length; i++) {
                if (shape[i] !== 1) {
                    newShape.push(shape[i]);
                    newStrides.push(strides[i]);
                }
            }
            return backend.createTensorHandle({
                storage: input.storage,
                shape: newShape.length > 0 ? newShape : [1],
                dtype: input.dtype,
                strides: newStrides.length > 0 ? newStrides : [1],
                offset: input.offset,
            });
        } else {
            const d = dim < 0 ? shape.length + dim : dim;
            if (shape[d] !== 1) {
                // cannot squeeze non-1 dim, return as-is
                return input;
            }
            shape.splice(d, 1);
            strides.splice(d, 1);

            if (shape.length === 0) {
                shape.push(1);
                strides.push(1);
            }

            return backend.createTensorHandle({
                storage: input.storage,
                shape,
                dtype: input.dtype,
                strides,
                offset: input.offset,
            });
        }
    }

    private flatten(
        input: ITensorHandle,
        startDim: number,
        endDim: number,
        backend: ReturnType<typeof env.getBackend>
    ): ITensorHandle {
        const ndim = input.shape.length;
        const start = startDim < 0 ? ndim + startDim : startDim;
        const end = endDim < 0 ? ndim + endDim : endDim;

        // 计算 flatten 后的维度
        let flatSize = 1;
        for (let i = start; i <= end; i++) {
            flatSize *= input.shape[i];
        }

        const newShape = [
            ...input.shape.slice(0, start),
            flatSize,
            ...input.shape.slice(end + 1),
        ];

        return this.reshape(input, newShape, backend);
    }

    private expand(
        input: ITensorHandle,
        size: number[],
        backend: ReturnType<typeof env.getBackend>
    ): ITensorHandle {
        // Expand 使用 stride trick: size-1 dims 扩展时 stride 设为 0
        const inputShape = input.shape;
        const inputStrides = input.strides;

        // 对齐到右边
        const ndimOut = size.length;
        const ndimIn = inputShape.length;
        const padDims = ndimOut - ndimIn;

        const newShape: number[] = [];
        const newStrides: number[] = [];

        for (let i = 0; i < ndimOut; i++) {
            const inIdx = i - padDims;
            const inDim = inIdx >= 0 ? inputShape[inIdx] : 1;
            const inStride = inIdx >= 0 ? inputStrides[inIdx] : 0;
            const outDim = size[i] === -1 ? inDim : size[i];

            if (inDim === 1 && outDim > 1) {
                // broadcast: stride = 0
                newShape.push(outDim);
                newStrides.push(0);
            } else if (inDim === outDim) {
                newShape.push(outDim);
                newStrides.push(inStride);
            } else {
                throw new Error(
                    `expand: cannot expand dim ${i} from ${inDim} to ${outDim}`
                );
            }
        }

        return backend.createTensorHandle({
            storage: input.storage,
            shape: newShape,
            dtype: input.dtype,
            strides: newStrides,
            offset: input.offset,
        });
    }

    /**
     * select - 沿指定维度选择单个索引（降维）
     * 
     * 返回一个降维的视图。例如对于 [2, 3] 的 tensor:
     * - select(0, 1) 返回 shape [3] (第二行)
     * - select(1, 0) 返回 shape [2] (第一列)
     */
    private select(
        input: ITensorHandle,
        dim: number,
        index: number,
        backend: ReturnType<typeof env.getBackend>
    ): ITensorHandle {
        const ndim = input.shape.length;
        const normalizedDim = dim < 0 ? ndim + dim : dim;

        if (normalizedDim < 0 || normalizedDim >= ndim) {
            throw new Error(`select: dim ${dim} out of range for tensor with ${ndim} dimensions`);
        }

        const dimSize = input.shape[normalizedDim];
        let normalizedIndex = index < 0 ? dimSize + index : index;

        if (normalizedIndex < 0 || normalizedIndex >= dimSize) {
            throw new Error(`select: index ${index} out of range for dim ${dim} with size ${dimSize}`);
        }

        // 计算新的 offset、shape、strides
        const newOffset = input.offset + normalizedIndex * (input.strides[normalizedDim] as number);

        const newShape: number[] = [];
        const newStrides: number[] = [];

        for (let i = 0; i < ndim; i++) {
            if (i !== normalizedDim) {
                newShape.push(input.shape[i]);
                newStrides.push(input.strides[i] as number);
            }
        }

        // 处理标量结果 (所有维度都被 select 掉)
        if (newShape.length === 0) {
            newShape.push(1);
            newStrides.push(1);
        }

        return backend.createTensorHandle({
            storage: input.storage,
            shape: newShape,
            dtype: input.dtype,
            strides: newStrides,
            offset: newOffset,
        });
    }

    /**
     * diagonal - 获取对角线视图
     * 
     * 行为匹配 PyTorch:
     * - 移除 dim1, dim2
     * - 在末尾添加对角线维度
     * - 调整 offset 和 计算新 stride
     */
    public diagonal(
        input: ITensorHandle,
        offset: number,
        dim1: number,
        dim2: number,
        backend: ReturnType<typeof env.getBackend>
    ): ITensorHandle {
        const off = offset ?? 0;
        const d_1 = dim1 ?? 0;
        const d_2 = dim2 ?? 1;

        const ndim = input.shape.length;
        const d1 = d_1 < 0 ? ndim + d_1 : d_1;
        const d2 = d_2 < 0 ? ndim + d_2 : d_2;

        if (d1 === d2) {
            throw new Error(`diagonal: dimensions must be distinct, got ${d1} and ${d2}`);
        }

        const size1 = input.shape[d1];
        const size2 = input.shape[d2];
        // Ensure strides are present and valid
        const stride1 = (input.strides[d1] ?? 1) as number;
        const stride2 = (input.strides[d2] ?? 1) as number;

        let diagSize: number;
        let newOffset = input.offset;

        if (off >= 0) {
            diagSize = Math.max(0, Math.min(size1, size2 - off));
            newOffset += off * stride2;
        } else {
            diagSize = Math.max(0, Math.min(size1 + off, size2));
            newOffset -= off * stride1;
        }

        const newShape: number[] = [];
        const newStrides: number[] = [];

        for (let i = 0; i < ndim; i++) {
            if (i !== d1 && i !== d2) {
                newShape.push(input.shape[i]);
                newStrides.push((input.strides[i] ?? 1) as number);
            }
        }

        newShape.push(diagSize);
        newStrides.push(stride1 + stride2);

        return backend.createTensorHandle({
            storage: input.storage,
            shape: newShape,
            dtype: input.dtype,
            strides: newStrides,
            offset: newOffset,
        });
    }

    /**
     * asStrided - 创建具有指定 size 和 stride 的视图
     * 
     * 对标 PyTorch: torch.as_strided(input, size, stride, storage_offset=None)
     * 
     * 这是一个强大但危险的操作，直接操作内存布局。
     * STFT 的分帧操作依赖此功能。
     * 
     * WARNING: 生成的视图必须只引用原张量存储中的元素
     */
    private asStrided(
        input: ITensorHandle,
        size: number[],
        stride: number[],
        storageOffset: number | undefined,
        backend: ReturnType<typeof env.getBackend>
    ): ITensorHandle {
        // 验证参数
        if (size.length !== stride.length) {
            throw new Error(
                `as_strided: size and stride must have the same length, ` +
                `got size=${size.length} and stride=${stride.length}`
            );
        }

        // 使用输入的 offset 如果 storageOffset 未指定
        const newOffset = storageOffset ?? input.offset;

        return backend.createTensorHandle({
            storage: input.storage,
            shape: size,
            dtype: input.dtype,
            strides: stride,
            offset: newOffset,
        });
    }

    /**
     * sliceByArrays - 内部切片操作 (零拷贝)
     * 
     * 接收已解析的 starts/ends/steps 数组，通过修改 offset 和 strides 实现切片。
     * 公共 API 的字符串解析在 execute 方法中完成。
     * 
     * 委托给 @kandle/utils/viewUtils.computeSliceParams
     */
    private sliceByArrays(
        input: ITensorHandle,
        starts: number[],
        ends: number[],
        steps: number[],
        backend: ReturnType<typeof env.getBackend>
    ): ITensorHandle {
        // 使用 viewUtils 中的 computeSliceParams 计算新的 shape/strides/offset
        const { newShape, newStrides, newOffset } = computeSliceParams(
            input.shape,
            input.strides as number[],
            input.offset,
            starts,
            ends,
            steps
        );

        return backend.createTensorHandle({
            storage: input.storage,
            shape: newShape,
            dtype: input.dtype,
            strides: newStrides,
            offset: newOffset,
        });
    }

    // ========================================================================
    // Cat Operations (Require Copy)
    // ========================================================================

    private executeCat(execCtx: DirectContext): ITensorHandle {
        const { inputs, scalars, metadata, kernelName, outs } = execCtx;

        // tensors 可能在 metadata.tensors (codegen 生成的调用) 或 inputs (直接调用)
        const tensors = (metadata?.['tensors'] as readonly ITensorHandle[] | undefined) ?? inputs;

        // 从 scalars 或 metadata 获取 dim 参数，默认为 0
        const dimValue = scalars?.['dim'] ?? metadata?.['dim'] ?? 0;
        const dim = typeof dimValue === 'number' ? dimValue : 0;

        switch (kernelName) {
            case 'cat':
                return this.cat(tensors, dim, outs?.[0]);

            case 'stack':
                return this.stack(tensors, dim, outs?.[0]);

            default:
                throw new Error(`Unknown cat operation: ${kernelName}`);
        }
    }

    /**
     * cat - 沿指定维度拼接张量序列
     *
     * @param tensors 输入张量数组
     * @param dim 拼接维度 (支持负数)
     * @param out 可选的输出张量
     */
    private cat(
        tensors: readonly ITensorHandle[],
        dim: number,
        out?: ITensorHandle
    ): ITensorHandle {
        if (tensors.length === 0) {
            throw new Error('cat: tensors must not be empty');
        }

        // 如果只有一个张量，直接返回 clone
        if (tensors.length === 1) {
            const backend = env.getBackend(tensors[0].device);
            return this.cloneTensor(tensors[0], backend);
        }

        const first = tensors[0];
        const ndim = first.shape.length;

        // 规范化 dim
        const normDim = dim < 0 ? ndim + dim : dim;
        if (normDim < 0 || normDim >= ndim) {
            throw new Error(
                `cat: dimension ${dim} out of range for ${ndim}D tensors`
            );
        }

        // 验证所有张量形状兼容
        let totalSize = first.shape[normDim];
        const outputShape = [...first.shape];

        for (let i = 1; i < tensors.length; i++) {
            const t = tensors[i];
            if (t.shape.length !== ndim) {
                throw new Error(
                    `cat: all tensors must have the same number of dimensions, ` +
                    `got ${ndim}D and ${t.shape.length}D`
                );
            }

            for (let d = 0; d < ndim; d++) {
                if (d !== normDim && t.shape[d] !== first.shape[d]) {
                    throw new Error(
                        `cat: incompatible shapes at dimension ${d}: ` +
                        `${first.shape[d]} vs ${t.shape[d]}`
                    );
                }
            }

            totalSize += t.shape[normDim];
        }

        outputShape[normDim] = totalSize;

        // 确定输出 dtype (promote)
        const outputDtype = this.promoteDtype(tensors);

        // 获取 backend
        const backend = env.getBackend(first.device);

        // 创建或验证输出张量
        let output: ITensorHandle;
        if (out) {
            // 验证 out 形状
            this.validateOutputShape(out, outputShape, 'cat');
            output = out;
        } else {
            output = backend.createTensorHandle(outputShape, outputDtype);
        }

        // 执行拼接
        this.catImpl(tensors, output, normDim, backend);

        return output;
    }

    /**
     * stack - 沿新维度堆叠张量序列
     *
     * @param tensors 输入张量数组
     * @param dim 新维度插入位置 (支持负数)
     * @param out 可选的输出张量
     */
    private stack(
        tensors: readonly ITensorHandle[],
        dim: number,
        out?: ITensorHandle
    ): ITensorHandle {
        if (tensors.length === 0) {
            throw new Error('stack: tensors must not be empty');
        }

        const first = tensors[0];
        const ndim = first.shape.length;

        // stack 创建一个新维度，所以 dim 范围是 [0, ndim]
        const normDim = dim < 0 ? ndim + 1 + dim : dim;
        if (normDim < 0 || normDim > ndim) {
            throw new Error(
                `stack: dimension ${dim} out of range for creating ${ndim + 1}D tensor`
            );
        }

        // 验证所有张量形状相同
        for (let i = 1; i < tensors.length; i++) {
            const t = tensors[i];
            if (t.shape.length !== ndim) {
                throw new Error(
                    `stack: all tensors must have the same number of dimensions, ` +
                    `got ${ndim}D and ${t.shape.length}D`
                );
            }

            for (let d = 0; d < ndim; d++) {
                if (t.shape[d] !== first.shape[d]) {
                    throw new Error(
                        `stack: all tensors must have the same shape, ` +
                        `but tensor 0 has shape [${first.shape}] and tensor ${i} has shape [${t.shape}]`
                    );
                }
            }
        }

        // 计算输出形状：在 normDim 位置插入 tensors.length
        const outputShape = [...first.shape];
        outputShape.splice(normDim, 0, tensors.length);

        // 确定输出 dtype
        const outputDtype = this.promoteDtype(tensors);

        // 获取 backend
        const backend = env.getBackend(first.device);

        // 创建或验证输出张量
        let output: ITensorHandle;
        if (out) {
            this.validateOutputShape(out, outputShape, 'stack');
            output = out;
        } else {
            output = backend.createTensorHandle(outputShape, outputDtype);
        }

        // 执行堆叠
        this.stackImpl(tensors, output, normDim, backend);

        return output;
    }

    /**
     * cat 的核心实现
     *
     * 策略：逐个张量使用 copy kernel 写入到输出的对应区域
     */
    private catImpl(
        tensors: readonly ITensorHandle[],
        output: ITensorHandle,
        dim: number,
        backend: ReturnType<typeof env.getBackend>
    ): void {
        const copyKernel = backend.operators.find('copy') as
            | IteratorKernelImpl
            | undefined;
        if (!copyKernel) {
            throw new Error('cat: copy kernel not found');
        }

        let currentOffset = 0; // 当前在 dim 维度上的偏移

        for (const tensor of tensors) {
            const sliceSize = tensor.shape[dim];

            // 创建输出的 slice view
            // 计算 slice 的 offset: currentOffset * output.strides[dim]
            const sliceOffset =
                output.offset + currentOffset * output.strides[dim];

            // 使用 backend.createTensorHandle 创建 slice view (共享 storage)
            const sliceView = backend.createTensorHandle({
                storage: output.storage,
                shape: [...tensor.shape], // slice 的形状与输入相同
                dtype: output.dtype,
                strides: [...output.strides], // 使用输出的 strides
                offset: sliceOffset,
            });

            // 使用 TensorIterator 复制数据
            const iter = TensorIterator.build({
                inputs: [tensor],
                outputs: [sliceView],
                opName: 'copy',
                disableDimensionCoalescing: true, // 防止维度折叠导致错误
            });

            copyKernel(iter);

            currentOffset += sliceSize;
        }
    }

    /**
     * stack 的核心实现
     *
     * 策略：将每个输入 unsqueeze 后复制到输出的对应位置
     */
    private stackImpl(
        tensors: readonly ITensorHandle[],
        output: ITensorHandle,
        dim: number,
        backend: ReturnType<typeof env.getBackend>
    ): void {
        const copyKernel = backend.operators.find('copy') as
            | IteratorKernelImpl
            | undefined;
        if (!copyKernel) {
            throw new Error('stack: copy kernel not found');
        }

        // 每个输入张量对应输出在 dim 维度上的一个 slice
        for (let i = 0; i < tensors.length; i++) {
            const tensor = tensors[i];

            // 计算输出 slice 的 offset
            // 在 dim 维度上，第 i 个位置
            const sliceOffset = output.offset + i * output.strides[dim];

            // 创建 slice view
            // 需要 "压缩" dim 维度 (因为输入没有这个维度)
            // 输出 strides 去掉 dim 位置的 stride
            const sliceStrides = [...output.strides];
            sliceStrides.splice(dim, 1);

            // 使用 backend.createTensorHandle 创建 slice view (共享 storage)
            const sliceView = backend.createTensorHandle({
                storage: output.storage,
                shape: [...tensor.shape], // 使用输入的形状 (无 dim 维度)
                dtype: output.dtype,
                strides: sliceStrides,
                offset: sliceOffset,
            });

            // 复制数据
            const iter = TensorIterator.build({
                inputs: [tensor],
                outputs: [sliceView],
                opName: 'copy',
                disableDimensionCoalescing: true,
            });

            copyKernel(iter);
        }
    }

    // ========================================================================
    // Utility Methods
    // ========================================================================

    private normalizeShape(shape: number[], numel: number): number[] {
        if (!shape) {
            throw new Error(`reshape: shape must be defined, got ${shape}`);
        }
        const negIdx = shape.indexOf(-1);
        if (negIdx === -1) {
            return shape;
        }

        // 计算 -1 对应的维度
        let product = 1;
        for (let i = 0; i < shape.length; i++) {
            if (i !== negIdx) {
                product *= shape[i];
            }
        }

        const inferredDim = numel / product;
        if (!Number.isInteger(inferredDim)) {
            throw new Error(`reshape: cannot infer dim for shape [${shape}] with numel ${numel}`);
        }

        const result = [...shape];
        result[negIdx] = inferredDim;
        return result;
    }

    private canView(input: ITensorHandle, newShape: number[]): boolean {
        // 检查是否 contiguous
        return this.isContiguous(input.shape, input.strides, input.offset);
    }

    private isContiguous(shape: readonly number[], strides: readonly number[], offset: number): boolean {
        if (offset !== 0) return false;

        let expectedStride = 1;
        for (let i = shape.length - 1; i >= 0; i--) {
            if (strides[i] !== expectedStride) {
                return false;
            }
            expectedStride *= shape[i];
        }
        return true;
    }
    /**
     * 类型提升：确定输出 dtype
     */
    private promoteDtype(tensors: readonly ITensorHandle[]): DType {
        // 简化实现：返回第一个张量的类型
        // 完整实现应该按照 PyTorch 的类型提升规则
        return tensors[0].dtype;
    }

    /**
     * 验证输出形状匹配
     */
    private validateOutputShape(
        out: ITensorHandle,
        expectedShape: readonly number[],
        opName: string
    ): void {
        if (out.shape.length !== expectedShape.length) {
            throw new Error(
                `${opName}: output tensor has wrong number of dimensions: ` +
                `expected ${expectedShape.length}, got ${out.shape.length}`
            );
        }

        for (let i = 0; i < expectedShape.length; i++) {
            if (out.shape[i] !== expectedShape[i]) {
                throw new Error(
                    `${opName}: output tensor has wrong shape at dimension ${i}: ` +
                    `expected ${expectedShape[i]}, got ${out.shape[i]}`
                );
            }
        }
    }

    /**
     * 克隆张量
     */
    private cloneTensor(
        input: ITensorHandle,
        backend: ReturnType<typeof env.getBackend>
    ): ITensorHandle {
        const output = backend.createTensorHandle([...input.shape], input.dtype);

        const iter = TensorIterator.unaryOp(input, 'copy', output);
        const copyKernel = backend.operators.find('copy') as
            | IteratorKernelImpl
            | undefined;
        if (copyKernel) {
            copyKernel(iter);
            return output;
        }

        throw new Error('Copy kernel not found');
    }
}

// ============================================================================
// Exports
// ============================================================================

// 主导出
export const dispatchShape = ShapeHandler.dispatch;

// View/Cat 操作的语义化导出 (都指向同一个 ShapeHandler)
export const dispatchView = ShapeHandler.dispatch;
export const dispatchCat = ShapeHandler.dispatch;
