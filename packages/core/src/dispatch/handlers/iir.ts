/**
 * IIR Handler
 *
 * 处理 IIR 滤波操作: lfilterBiquad
 *
 * Mechanism: Factory (自定义 handler 创建输出 tensor)
 */

import type { ITensorHandle, DType, DeviceNameEnum } from '@kandle/types';
import type { OpEntry } from '@kandle/types';
import { env } from '../../env';
import type { PatternHandler, OperatorContext, DirectContext } from './types';

/**
 * IIR Biquad Kernel Args
 */
export interface IIRBiquadKernelArgs {
    input: ITensorHandle;
    output: ITensorHandle;
    b0: number;
    b1: number;
    b2: number;
    a1: number;
    a2: number;
    clamp: boolean;
    clampMin: number;
    clampMax: number;
}

export class IIRHandler implements PatternHandler {
    private static instance: IIRHandler;

    static getInstance(): IIRHandler {
        if (!IIRHandler.instance) {
            IIRHandler.instance = new IIRHandler();
        }
        return IIRHandler.instance;
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

        const input = inputs[0];
        if (!input) {
            throw new Error('IIR: input tensor is required');
        }

        // 提取系数
        const b0 = allParams['b0'] as number;
        const b1 = allParams['b1'] as number;
        const b2 = allParams['b2'] as number;
        const a1 = allParams['a1'] as number;
        const a2 = allParams['a2'] as number;
        const clamp = (allParams['clamp'] ?? true) as boolean;

        // 确保输入是连续的 (IIR kernel 需要连续内存)
        // 注: 这里应该调用 contiguous 操作，但需要通过 dispatch 链路
        // 暂时假设输入已经连续或由调用方处理
        const inputHandle = input;

        // 创建输出 tensor (形状与输入相同)
        const shape = [...inputHandle.shape];
        const dtype = inputHandle.dtype as DType;
        const output = backend.createTensorHandle(shape, dtype);

        // 获取 IIR kernel
        const kernel = backend.operators.find('iir.biquad');
        if (!kernel) {
            throw new Error('IIR biquad kernel not available on backend');
        }

        // 构建 kernel args
        const args: IIRBiquadKernelArgs = {
            input: inputHandle,
            output,
            b0,
            b1,
            b2,
            a1,
            a2,
            clamp,
            clampMin: -1.0,
            clampMax: 1.0,
        };

        // 构建 DirectContext 并调用 kernel
        const kernelCtx: DirectContext = {
            kind: 'direct',
            inputs: [inputHandle],
            scalars: args as any,
            metadata: {},
            outs: [output],
            kernelName: 'iir.biquad',
        };

        (kernel as (ctx: DirectContext) => void)(kernelCtx);

        return output;
    }

    static dispatch(entry: OpEntry, ctx: OperatorContext): ITensorHandle {
        const handler = IIRHandler.getInstance();
        const execCtx = handler.buildContext(entry, ctx);
        return handler.execute(execCtx);
    }
}

export const dispatchIIR = IIRHandler.dispatch;
