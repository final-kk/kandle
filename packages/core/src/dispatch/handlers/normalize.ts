/**
 * v5 NormalizeHandler
 *
 * 处理归一化操作: softmax, log_softmax, softmin, layer_norm, batch_norm, group_norm, rms_norm, normalize
 * 
 * 特点:
 * - 沿指定维度计算统计量 (max, sum, mean, var 等)
 * - 广播回原形状应用归一化
 * - 输出形状等于输入形状
 */

import type { ITensorHandle } from '@kandle/types';
import type { OpEntry } from '@kandle/types';
import type { PatternHandler, OperatorContext, DirectContext } from './types';
import { env } from '../../env';

/**
 * NormalizeHandler - 处理 Normalize 计算模式
 *
 * 覆盖算子: softmax, log_softmax, softmin,
 *          layer_norm, batch_norm, group_norm, rms_norm, normalize
 */
export class NormalizeHandler implements PatternHandler {
    private static instance: NormalizeHandler;

    static getInstance(): NormalizeHandler {
        if (!NormalizeHandler.instance) {
            NormalizeHandler.instance = new NormalizeHandler();
        }
        return NormalizeHandler.instance;
    }

    buildContext(entry: OpEntry, ctx: OperatorContext): DirectContext {
        const { tensorInputs, scalarArgs, metadata, outs } = ctx;

        return {
            kind: 'direct',
            inputs: tensorInputs,
            scalars: scalarArgs,
            metadata,
            outs,
            kernelName: entry.dispatchKey,
        };
    }

    execute(execCtx: DirectContext): ITensorHandle {
        const { inputs, scalars, metadata, outs, kernelName } = execCtx;

        const device = inputs[0].device;
        const backend = env.getBackend(device);

        // 查找 normalize kernel
        const kernel = backend.operators.find(kernelName) as (
            (inputs: ITensorHandle[], params: Record<string, unknown>) => ITensorHandle
        ) | undefined;

        if (!kernel) {
            throw new Error(`Kernel '${kernelName}' not found for device '${device}'`);
        }

        // 构建 kernel params - 需要将 tensor inputs 映射到命名参数
        const params: Record<string, unknown> = { ...scalars, ...metadata, out: outs?.[0] };

        // 根据 kernelName 解析 tensor 参数
        // inputs[0] 始终是主输入 tensor
        switch (kernelName) {
            case 'batch_norm':
                // batchNorm: [self, runningMean?, runningVar?, weight?, bias?]
                if (inputs.length > 1) params.runningMean = inputs[1];
                if (inputs.length > 2) params.runningVar = inputs[2];
                if (inputs.length > 3) params.weight = inputs[3];
                if (inputs.length > 4) params.bias = inputs[4];
                break;

            case 'layer_norm':
            case 'rms_norm':
                // layerNorm/rmsNorm: [self, weight?, bias?]
                if (inputs.length > 1) params.weight = inputs[1];
                if (inputs.length > 2) params.bias = inputs[2];
                break;

            case 'group_norm':
                // groupNorm: [self, weight?, bias?]
                if (inputs.length > 1) params.weight = inputs[1];
                if (inputs.length > 2) params.bias = inputs[2];
                break;

            // softmax, log_softmax, softmin, lp_normalize 只有一个输入
            default:
                break;
        }

        // 调用 kernel - 只传递主输入
        return kernel([inputs[0]], params);
    }

    /**
     * 一键执行: build + execute
     */
    static dispatch(entry: OpEntry, ctx: OperatorContext): ITensorHandle {
        const handler = NormalizeHandler.getInstance();
        const execCtx = handler.buildContext(entry, ctx);
        return handler.execute(execCtx);
    }
}

// 便捷函数导出
export const dispatchNormalize = NormalizeHandler.dispatch;
