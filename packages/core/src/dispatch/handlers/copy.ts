/**
 * v5 CopyHandler
 *
 * 处理数据复制操作: contiguous, clone, to
 */

import type { ITensorHandle, DType, DeviceNameEnum } from '@kandle/types';
import { MemoryFormat } from '@kandle/types';
import type { OpEntry } from '@kandle/types';
import type { IteratorKernelImpl } from '@kandle/types';
import { computeStridesForFormat, inferMemoryFormat } from '@kandle/utils';
import { env } from '../../env';
import { TensorIterator } from '../TensorIterator';
import type { PatternHandler, OperatorContext, DirectContext } from './types';

export class CopyHandler implements PatternHandler {
    private static instance: CopyHandler;

    static getInstance(): CopyHandler {
        if (!CopyHandler.instance) {
            CopyHandler.instance = new CopyHandler();
        }
        return CopyHandler.instance;
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
        const input = inputs[0];
        const backend = env.getBackend(input.device);

        switch (kernelName) {
            case 'contiguous': {
                const memoryFormat = allParams['memoryFormat'] as MemoryFormat | undefined;
                return this.contiguous(input, memoryFormat, backend);
            }

            case 'clone':
                return this.clone(input, backend);

            case 'cast': {
                const targetDtype = allParams['dtype'] as DType;
                return this.cast(input, targetDtype, backend);
            }

            case 'to': {
                const targetDevice = allParams['device'] as DeviceNameEnum | undefined;
                const targetDtype = allParams['dtype'] as DType | undefined;
                return this.to(input, targetDevice, targetDtype);
            }

            case 'copy_': {
                const src = inputs[1];
                return this.copyInplace(input, src, backend);
            }

            default:
                throw new Error(`Unknown copy operation: ${kernelName}`);
        }
    }

    static dispatch(entry: OpEntry, ctx: OperatorContext): ITensorHandle {
        const handler = CopyHandler.getInstance();
        const execCtx = handler.buildContext(entry, ctx);
        return handler.execute(execCtx);
    }

    /**
     * 检查 tensor 是否已经是指定的 memoryFormat
     * 
     * 对于 Contiguous 格式，除了 format 匹配外还需要验证物理连续性，
     * 因为 inferMemoryFormat 对低维 tensor 总是返回 Contiguous。
     */
    private isTargetFormat(input: ITensorHandle, targetFormat: MemoryFormat): boolean {
        if (input.offset !== 0) return false;

        const currentFormat = inferMemoryFormat(input.shape, input.strides);

        // 对于 Contiguous 格式，需要额外检查物理连续性
        // 因为 inferMemoryFormat 对 <4D tensor 始终返回 Contiguous
        if (targetFormat === MemoryFormat.Contiguous) {
            return this.isContiguous(input);
        }

        return currentFormat === targetFormat;
    }

    /**
     * 检查 tensor 是否为标准连续格式 (row-major, offset=0)
     */
    private isContiguous(input: ITensorHandle): boolean {
        const { shape, strides, offset } = input;
        if (offset !== 0) return false;
        let expectedStride = 1;
        for (let i = shape.length - 1; i >= 0; i--) {
            if (strides[i] !== expectedStride) return false;
            expectedStride *= shape[i];
        }
        return true;
    }

    /**
     * contiguous - 确保 tensor 按指定格式连续存储
     * 
     * 支持 MemoryFormat:
     * - Contiguous: 标准 row-major (NCHW for 4D)
     * - ChannelsLast: 通道最密集 (NHWC physical layout for 4D)
     * - ChannelsLast3d: 5D 版本
     * 
     * 如果已是目标格式，返回原 tensor (zero-copy)。
     */
    private contiguous(
        input: ITensorHandle,
        memoryFormat: MemoryFormat | undefined,
        backend: ReturnType<typeof env.getBackend>
    ): ITensorHandle {
        // 解析 memoryFormat 字符串为 enum (如果需要)
        const targetFormat = this.resolveMemoryFormat(memoryFormat);

        // 检查是否已经是目标格式
        if (this.isTargetFormat(input, targetFormat)) {
            return input;
        }

        // 需要转换：创建目标格式的输出 tensor 并拷贝
        return this.cloneWithFormat(input, targetFormat, backend);
    }

    /**
     * 解析 memoryFormat 参数
     * 
     * 支持 string 和 MemoryFormat enum 两种形式
     */
    private resolveMemoryFormat(format: MemoryFormat | string | undefined): MemoryFormat {
        if (!format) {
            return 'contiguous' as MemoryFormat;
        }
        // MemoryFormat enum values are strings: 'contiguous', 'channels_last', etc.
        return format as MemoryFormat;
    }

    /**
     * 克隆 tensor 到指定的内存格式
     * 
     * 关键点：当输入输出 strides 布局不同时（如 NCHW -> NHWC），
     * 必须禁用维度折叠优化，否则会错误地合并具有不同 strides 模式的维度。
     */
    private cloneWithFormat(
        input: ITensorHandle,
        targetFormat: MemoryFormat,
        backend: ReturnType<typeof env.getBackend>
    ): ITensorHandle {
        // 计算目标格式的 strides
        const targetStrides = computeStridesForFormat(input.shape, targetFormat);

        // 创建输出 tensor
        const output = backend.createTensorHandle({
            shape: [...input.shape],
            dtype: input.dtype,
            strides: targetStrides,
            memoryFormat: targetFormat,
        });

        // 使用 TensorIterator.build 并禁用维度折叠
        // 因为输入和输出具有不同的内存布局，维度折叠会导致错误的索引计算
        const iter = TensorIterator.build({
            inputs: [input],
            outputs: [output],
            opName: 'copy',
            disableDimensionCoalescing: true,  // <-- 关键：禁用维度折叠
        });

        const copyKernel = backend.operators.find('copy') as IteratorKernelImpl | undefined;
        if (copyKernel) {
            copyKernel(iter);
            return output;
        }

        throw new Error('Copy kernel not found');
    }

    private clone(input: ITensorHandle, backend: ReturnType<typeof env.getBackend>): ITensorHandle {
        // 创建输出
        const output = backend.createTensorHandle([...input.shape], input.dtype);

        // 使用 TensorIterator + copy kernel
        const iter = TensorIterator.unaryOp(input, 'copy', output);
        const copyKernel = backend.operators.find('copy') as IteratorKernelImpl | undefined;
        if (copyKernel) {
            copyKernel(iter);
            return output;
        }

        throw new Error('Copy kernel not found');
    }

    /**
     * cast - 纯类型转换
     * 
     * 设计原则:
     * - 如果 dtype 相同，返回原 tensor (zero-copy)
     * - 否则创建新 tensor 并使用 copy kernel 进行转换
     * - copy kernel 天然支持类型转换 (读取源 dtype，写入目标 dtype)
     */
    private cast(
        input: ITensorHandle,
        targetDtype: DType,
        backend: ReturnType<typeof env.getBackend>
    ): ITensorHandle {
        // Zero-copy optimization: 如果 dtype 相同，直接返回
        if (input.dtype === targetDtype) {
            return input;
        }

        // 创建目标 dtype 的输出 tensor
        const output = backend.createTensorHandle([...input.shape], targetDtype);

        // 使用 TensorIterator + copy kernel
        // copy kernel 天然支持类型转换:
        // - 读取时使用 input.dtype
        // - 写入时使用 output.dtype (即 targetDtype)
        const iter = TensorIterator.unaryOp(input, 'copy', output);
        const copyKernel = backend.operators.find('copy') as IteratorKernelImpl | undefined;
        if (copyKernel) {
            copyKernel(iter);
            return output;
        }

        throw new Error('Copy kernel not found');
    }

    private to(
        input: ITensorHandle,
        targetDevice: DeviceNameEnum | undefined,
        targetDtype: DType | undefined
    ): ITensorHandle {
        const device = targetDevice ?? input.device;
        const dtype = targetDtype ?? input.dtype;

        if (device === input.device && dtype === input.dtype) {
            return input; // No change needed
        }

        const backend = env.getBackend(device);
        const output = backend.createTensorHandle([...input.shape], dtype);

        // 使用 TensorIterator + cast/copy kernel
        const iter = TensorIterator.unaryOp(input, 'copy', output);
        const copyKernel = backend.operators.find('copy') as IteratorKernelImpl | undefined;
        if (copyKernel) {
            copyKernel(iter);
            return output;
        }

        throw new Error('Copy kernel not found');
    }

    /**
     * copy_ - 原地拷贝
     *
     * 将 src 的内容拷贝到 self (in-place)。
     * self 可以是非连续的 view (如 slice 返回的 view)。
     * 
     * 关键点：
     * - output 是 self (目标 tensor，可能非连续)
     * - input 是 src (源 tensor)
     * - 必须禁用维度折叠，因为 self 可能有非标准 strides
     */
    private copyInplace(
        self: ITensorHandle,
        src: ITensorHandle,
        backend: ReturnType<typeof env.getBackend>
    ): ITensorHandle {
        // 验证形状兼容性
        if (self.shape.length !== src.shape.length) {
            throw new Error(
                `copy_: shape mismatch - self has ${self.shape.length} dims, ` +
                `src has ${src.shape.length} dims`
            );
        }
        for (let i = 0; i < self.shape.length; i++) {
            if (self.shape[i] !== src.shape[i]) {
                throw new Error(
                    `copy_: shape mismatch at dim ${i} - self[${self.shape[i]}] vs src[${src.shape[i]}]`
                );
            }
        }

        // 使用 TensorIterator.build 并禁用维度折叠
        // self 作为 output，src 作为 input
        const iter = TensorIterator.build({
            inputs: [src],
            outputs: [self],
            opName: 'copy',
            disableDimensionCoalescing: true,  // self 可能是非连续 view
        });

        const copyKernel = backend.operators.find('copy') as IteratorKernelImpl | undefined;
        if (copyKernel) {
            copyKernel(iter);
            return self;  // 返回修改后的 self
        }

        throw new Error('Copy kernel not found');
    }
}

export const dispatchCopy = CopyHandler.dispatch;
