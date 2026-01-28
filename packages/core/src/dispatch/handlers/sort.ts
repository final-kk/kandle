/**
 * v5 SortHandler
 *
 * 处理排序操作: sort, argsort, topk
 * 
 * 参考 PyTorch 实现:
 * - 使用专用 kernel，不使用 TensorIterator
 * - sort/topk 返回 (values, indices) tuple
 * - argsort 只返回 indices
 * - 支持参数: dim, descending, stable (sort/argsort), largest, sorted (topk)
 */

import type { ITensorHandle } from '@kandle/types';
import type { OpEntry } from '@kandle/types';
import { env } from '../../env';
import type {
    PatternHandler,
    OperatorContext,
    DirectContext,
} from './types';

export class SortHandler implements PatternHandler {
    private static instance: SortHandler;

    static getInstance(): SortHandler {
        if (!SortHandler.instance) {
            SortHandler.instance = new SortHandler();
        }
        return SortHandler.instance;
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

    execute(execCtx: DirectContext): ITensorHandle | ITensorHandle[] {
        const { inputs, scalars, metadata, outs, kernelName } = execCtx;

        const backend = env.getBackend(inputs[0].device);
        const kernel = backend.operators.find(kernelName);

        if (!kernel) {
            throw new Error(
                `Sort kernel '${kernelName}' not implemented for backend '${inputs[0].device}'. ` +
                `Sort operations (sort, argsort, topk) require backend-specific kernel implementations.`
            );
        }

        // Sort kernel 的调用约定:
        // 不同于 TensorIterator-based kernels, Sort kernels 接收配置对象并返回结果
        // - topk/sort 返回 [values, indices] tuple
        // - argsort 返回 indices tensor

        // Convert scalars to Record (kernel expects Record<string, unknown>)
        // Use Object.assign for O(1) property access and merging
        const scalarsRecord: Record<string, unknown> = { ...scalars };

        // Also merge metadata into scalars for parameters like 'dim'
        if (metadata) {
            Object.assign(scalarsRecord, metadata);
        }

        const result = (kernel as any)(
            inputs[0],
            scalarsRecord,
            outs
        ) as ITensorHandle | ITensorHandle[];

        return result;
    }

    static dispatch(entry: OpEntry, ctx: OperatorContext): ITensorHandle | ITensorHandle[] {
        const handler = SortHandler.getInstance();
        const execCtx = handler.buildContext(entry, ctx);
        return handler.execute(execCtx);
    }
}

export const dispatchSort = SortHandler.dispatch;
