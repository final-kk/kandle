/**
 * v5 ScatterHandler
 *
 * 处理散射/索引写入操作: scatter, scatter_add, scatter_reduce
 * 
 * Scatter pattern 特点:
 * - 基于索引张量将源张量元素写入目标张量
 * - scatter_add/scatter_reduce 需要原子操作处理写冲突
 * - 需要专用 kernel，不使用 TensorIterator
 */

import type { ITensorHandle } from '@kandle/types';
import type { OpEntry } from '@kandle/types';
import { env } from '../../env';
import type { PatternHandler, OperatorContext, DirectContext } from './types';

/**
 * Scatter Kernel 的函数签名
 */
type ScatterKernelImpl = (
    self: ITensorHandle,
    index: ITensorHandle,
    src: ITensorHandle,
    params: Record<string, unknown>,
    output: ITensorHandle
) => void;

/**
 * Scatter 归约模式
 */
export type ScatterReduceMode = 'sum' | 'prod' | 'mean' | 'amax' | 'amin';

export class ScatterHandler implements PatternHandler {
    private static instance: ScatterHandler;

    static getInstance(): ScatterHandler {
        if (!ScatterHandler.instance) {
            ScatterHandler.instance = new ScatterHandler();
        }
        return ScatterHandler.instance;
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
            case 'scatter': {
                const self = inputs[0];
                const index = inputs[1];
                const src = inputs[2];
                const dim = this.normalizeDim(allParams['dim'] as number, self.shape.length);
                return this.scatter(self, dim, index, src, out);
            }

            case 'scatter_add': {
                const self = inputs[0];
                const index = inputs[1];
                const src = inputs[2];
                const dim = this.normalizeDim(allParams['dim'] as number, self.shape.length);
                return this.scatterAdd(self, dim, index, src, out);
            }

            case 'scatter_reduce': {
                const self = inputs[0];
                const index = inputs[1];
                const src = inputs[2];
                const dim = this.normalizeDim(allParams['dim'] as number, self.shape.length);
                const reduce = allParams['reduce'] as ScatterReduceMode;
                const includeSelf = allParams['includeSelf'] as boolean ?? true;
                return this.scatterReduce(self, dim, index, src, reduce, includeSelf, out);
            }

            default:
                throw new Error(`Unknown scatter operation: ${kernelName}`);
        }
    }

    static dispatch(entry: OpEntry, ctx: OperatorContext): ITensorHandle {
        const handler = ScatterHandler.getInstance();
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
     * 验证 scatter 操作的输入参数
     */
    private validateScatterInputs(
        self: ITensorHandle,
        dim: number,
        index: ITensorHandle,
        src: ITensorHandle
    ): void {
        const selfNdim = self.shape.length;
        const indexNdim = index.shape.length;
        const srcNdim = src.shape.length;

        // index 和 src 必须有相同的维度数
        if (indexNdim !== srcNdim) {
            throw new Error(
                `scatter: index and src must have the same number of dimensions, ` +
                `got index.ndim=${indexNdim}, src.ndim=${srcNdim}`
            );
        }

        // self 和 src 也必须有相同的维度数 (PyTorch 要求)
        if (selfNdim !== srcNdim) {
            throw new Error(
                `scatter: self and src must have the same number of dimensions, ` +
                `got self.ndim=${selfNdim}, src.ndim=${srcNdim}`
            );
        }

        // 验证尺寸约束
        for (let d = 0; d < selfNdim; d++) {
            if (index.shape[d] > src.shape[d]) {
                throw new Error(
                    `scatter: index.size(${d}) = ${index.shape[d]} is greater than ` +
                    `src.size(${d}) = ${src.shape[d]}`
                );
            }
            if (d !== dim && index.shape[d] > self.shape[d]) {
                throw new Error(
                    `scatter: index.size(${d}) = ${index.shape[d]} is greater than ` +
                    `self.size(${d}) = ${self.shape[d]} (dim=${dim})`
                );
            }
        }
    }

    /**
     * scatter - 沿指定维度将 src 值写入 self 的指定位置
     * 
     * PyTorch 语义:
     * - out = self.clone()
     * - out[index[i][j][k]][j][k] = src[i][j][k]  (dim=0)
     * 
     * @param self 目标张量
     * @param dim 散射维度 (已标准化)
     * @param index 索引张量
     * @param src 源张量
     * @param out 可选的输出张量
     */
    private scatter(
        self: ITensorHandle,
        dim: number,
        index: ITensorHandle,
        src: ITensorHandle,
        out?: ITensorHandle
    ): ITensorHandle {
        this.validateScatterInputs(self, dim, index, src);

        const backend = env.getBackend(self.device);

        // 创建或验证输出张量
        // 输出形状与 self 相同
        const output = out ?? backend.createTensorHandle({
            shape: [...self.shape],
            dtype: self.dtype,
        });

        // 调用 backend kernel
        const scatterKernel = backend.operators.find('scatter');
        if (!scatterKernel) {
            throw new Error('scatter kernel not found');
        }

        // 调用 kernel
        (scatterKernel as unknown as ScatterKernelImpl)(self, index, src, { dim }, output);

        return output;
    }

    /**
     * scatter_add - 沿指定维度将 src 值原子累加到 self 的指定位置
     * 
     * @param self 目标张量 (初始值)
     * @param dim 散射维度 (已标准化)
     * @param index 索引张量
     * @param src 源张量
     * @param out 可选的输出张量
     */
    private scatterAdd(
        self: ITensorHandle,
        dim: number,
        index: ITensorHandle,
        src: ITensorHandle,
        out?: ITensorHandle
    ): ITensorHandle {
        this.validateScatterInputs(self, dim, index, src);

        const backend = env.getBackend(self.device);

        // 创建输出张量
        const output = out ?? backend.createTensorHandle({
            shape: [...self.shape],
            dtype: self.dtype,
        });

        // 调用 backend kernel
        const scatterAddKernel = backend.operators.find('scatter_add');
        if (!scatterAddKernel) {
            throw new Error('scatter_add kernel not found');
        }

        (scatterAddKernel as unknown as ScatterKernelImpl)(self, index, src, { dim }, output);

        return output;
    }

    /**
     * scatter_reduce - 通用散射归约操作
     * 
     * @param self 目标张量
     * @param dim 散射维度 (已标准化)
     * @param index 索引张量
     * @param src 源张量
     * @param reduce 归约模式
     * @param includeSelf 是否包含 self 原值
     * @param out 可选的输出张量
     */
    private scatterReduce(
        self: ITensorHandle,
        dim: number,
        index: ITensorHandle,
        src: ITensorHandle,
        reduce: ScatterReduceMode,
        includeSelf: boolean,
        out?: ITensorHandle
    ): ITensorHandle {
        this.validateScatterInputs(self, dim, index, src);

        const backend = env.getBackend(self.device);

        // 创建输出张量
        const output = out ?? backend.createTensorHandle({
            shape: [...self.shape],
            dtype: self.dtype,
        });

        // 调用 backend kernel
        const scatterReduceKernel = backend.operators.find('scatter_reduce');
        if (!scatterReduceKernel) {
            throw new Error('scatter_reduce kernel not found');
        }

        (scatterReduceKernel as unknown as ScatterKernelImpl)(self, index, src, { dim, reduce, includeSelf }, output);

        return output;
    }
}

export const dispatchScatter = ScatterHandler.dispatch;
