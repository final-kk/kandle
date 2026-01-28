/**
 * v5 GatherHandler
 *
 * 处理索引选择操作: index_select, gather, embedding 等
 * 
 * Gather pattern 特点:
 * - 基于索引张量读取源张量元素
 * - 需要专用 kernel，不使用 TensorIterator
 */

import type { ITensorHandle, IteratorKernelImpl } from '@kandle/types';
import type { OpEntry } from '@kandle/types';
import { env } from '../../env';
import { computeStrides } from '@kandle/utils';
import { TensorIterator } from '../TensorIterator';
import type { PatternHandler, OperatorContext, DirectContext } from './types';

export class GatherHandler implements PatternHandler {
    private static instance: GatherHandler;

    static getInstance(): GatherHandler {
        if (!GatherHandler.instance) {
            GatherHandler.instance = new GatherHandler();
        }
        return GatherHandler.instance;
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
        const { inputs, scalars, metadata, outs, kernelName } = execCtx;
        const out = outs?.[0];
        const allParams = { ...scalars, ...metadata };

        switch (kernelName) {
            case 'index_select': {
                const self = inputs[0];
                const index = inputs[1];
                const dim = this.normalizeDim(allParams['dim'] as number, self.shape.length);
                return this.indexSelect(self, dim, index, out);
            }

            case 'embedding': {
                // F.embedding: output[...] = weight[input[...], :]
                // inputs[0] = input (indices, any shape, int32/int64)
                // inputs[1] = weight (vocab_size, embed_dim)
                const input = inputs[0];
                const weight = inputs[1];
                return this.embedding(input, weight, out);
            }

            default:
                throw new Error(`Unknown gather operation: ${kernelName}`);
        }
    }

    static dispatch(entry: OpEntry, ctx: OperatorContext): ITensorHandle {
        const handler = GatherHandler.getInstance();
        const execCtx = handler.buildContext(entry, ctx);
        return handler.execute(execCtx);
    }

    /**
     * 标准化维度索引 (支持负数)
     */
    private normalizeDim(dim: number, ndim: number): number {
        if (dim < 0) {
            dim = ndim + dim;
        }
        if (dim < 0 || dim >= ndim) {
            throw new Error(`Dimension out of range (expected to be in range of [${-ndim}, ${ndim - 1}], but got ${dim})`);
        }
        return dim;
    }

    /**
     * index_select - 沿指定维度使用索引张量选择元素
     * 
     * PyTorch 语义:
     * - output.shape = self.shape with self.shape[dim] replaced by index.shape[0]
     * - output[..., i, ...] = self[..., index[i], ...]  (i-th position is dim)
     * 
     * @param self 源张量
     * @param dim 选择维度 (已标准化)
     * @param index 1D 索引张量 (int32 或 int64)
     * @param out 可选的输出张量
     */
    private indexSelect(
        self: ITensorHandle,
        dim: number,
        index: ITensorHandle,
        out?: ITensorHandle
    ): ITensorHandle {
        const backend = env.getBackend(self.device);

        // 验证 index 是 1D
        if (index.shape.length !== 1) {
            throw new Error(`index_select: index must be 1D tensor, got ${index.shape.length}D`);
        }

        // 计算输出形状
        const outputShape = [...self.shape];
        outputShape[dim] = index.shape[0];

        // 创建或验证输出张量
        const output = out ?? backend.createTensorHandle({
            shape: outputShape,
            dtype: self.dtype,
        });

        // 调用 backend kernel
        const indexSelectKernel = backend.operators.find('index_select');
        if (!indexSelectKernel) {
            throw new Error('index_select kernel not found');
        }

        // 调用 kernel
        // GatherKernelImpl 签名: (self, index, params, output) -> void
        (indexSelectKernel as (
            self: ITensorHandle,
            index: ITensorHandle,
            params: Record<string, unknown>,
            output: ITensorHandle
        ) => void)(self, index, { dim }, output);

        return output;
    }

    /**
     * embedding - 嵌入查找
     * 
     * 分解为:
     * 1. flatten input indices to 1D: flat_indices = input.flatten()
     * 2. index_select: selected = index_select(weight, 0, flat_indices)
     * 3. view: output = selected.view(input.shape + [embed_dim])
     * 
     * PyTorch 语义:
     * - output[...] = weight[input[...], :]
     * - output.shape = input.shape + [embed_dim]
     * 
     * @param input 索引张量 (任意形状，int32/int64)
     * @param weight 嵌入矩阵 (vocab_size, embed_dim)
     * @param out 可选的输出张量
     */
    private embedding(
        input: ITensorHandle,
        weight: ITensorHandle,
        out?: ITensorHandle
    ): ITensorHandle {
        const backend = env.getBackend(input.device);
        const inputShape = input.shape;
        const embedDim = weight.shape[1];

        // 计算输出形状: input.shape + [embed_dim]
        const outputShape = [...inputShape, embedDim];

        // Step 1: Flatten input to 1D
        // 通过 view 创建扁平化的索引视图 (零拷贝)
        const numIndices = inputShape.reduce((a, b) => a * b, 1);
        const flatIndices = backend.createTensorHandle({
            shape: [numIndices],
            dtype: input.dtype,
            storage: input.storage,
            strides: [1],  // 假设 input 是连续的，或者需要先 contiguous
            offset: input.offset,
        });

        // 如果 input 不是连续的，需要先做 contiguous
        // 这里简化处理，假设 input 是连续的
        // TODO: 如果需要，添加 contiguous 检查

        // Step 2: index_select(weight, 0, flat_indices)
        // 输出形状: [numIndices, embedDim]
        const selectedShape = [numIndices, embedDim];
        const selected = backend.createTensorHandle({
            shape: selectedShape,
            dtype: weight.dtype,
        });

        // 调用 index_select kernel
        const indexSelectKernel = backend.operators.find('index_select');
        if (!indexSelectKernel) {
            throw new Error('index_select kernel not found (required for embedding)');
        }

        (indexSelectKernel as (
            self: ITensorHandle,
            index: ITensorHandle,
            params: Record<string, unknown>,
            output: ITensorHandle
        ) => void)(weight, flatIndices, { dim: 0 }, selected);

        // Step 3: View selected to output shape
        // selected shape: [numIndices, embedDim] -> view to input.shape + [embedDim]
        // 由于 selected 是连续的，可以直接创建 view
        const viewedResult = backend.createTensorHandle({
            shape: outputShape,
            dtype: weight.dtype,
            storage: selected.storage,
            strides: computeStrides(outputShape),
            offset: 0,
        });

        // 如果提供了 out，使用 copy kernel 复制结果
        if (out) {
            const copyIter = TensorIterator.unaryOp(viewedResult, 'copy', out);
            const copyKernel = backend.operators.find('copy') as IteratorKernelImpl | undefined;
            if (!copyKernel) throw new Error("Kernel 'copy' not found");
            copyKernel(copyIter);
            return out;
        }

        return viewedResult;
    }
}

export const dispatchGather = GatherHandler.dispatch;

