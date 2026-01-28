/**
 * WindowFunc Handler
 *
 * 处理窗函数生成操作: hann_window, hamming_window, blackman_window, bartlett_window, kaiser_window
 * 
 * Mechanism: WindowFunc
 */

import type { ITensorHandle, DType, DeviceNameEnum } from '@kandle/types';
import type { OpEntry } from '@kandle/types';
import type { WindowFuncKernelArgs, GeneralizedCosineConfig, LinearConfig, KaiserConfig } from '@kandle/backend-webgpu';
import { env } from '../../env';
import type { PatternHandler, OperatorContext, DirectContext } from './types';

/**
 * 计算窗函数的 denominator 参数
 * 
 * @param N 窗口长度
 * @param options periodic 或 sym 参数
 * @returns denominator 值
 */
function computeDenominator(N: number, options: { periodic?: boolean; sym?: boolean }): number {
    // Top-level API: periodic 参数
    if (options.periodic !== undefined) {
        return options.periodic ? N : N - 1;
    }

    // Signal.Windows API: sym 参数
    if (options.sym !== undefined) {
        return options.sym ? N - 1 : N;
    }

    // 默认值: symmetric
    return N - 1;
}

export class WindowFuncHandler implements PatternHandler {
    private static instance: WindowFuncHandler;

    static getInstance(): WindowFuncHandler {
        if (!WindowFuncHandler.instance) {
            WindowFuncHandler.instance = new WindowFuncHandler();
        }
        return WindowFuncHandler.instance;
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
        const { scalars, metadata, kernelName } = execCtx;
        const allParams = { ...scalars, ...metadata };
        const device = (allParams['device'] as DeviceNameEnum) ?? env.getDefaultDevice().name;
        const backend = env.getBackend(device);
        const dtype = (allParams['dtype'] ?? 'float32') as DType;

        // 获取通用参数
        const windowLength = allParams['windowLength'] as number;
        const periodic = allParams['periodic'] as boolean | undefined;
        const sym = allParams['sym'] as boolean | undefined;
        const denominator = computeDenominator(windowLength, { periodic, sym });

        // 创建输出 tensor
        const shape = [windowLength];
        const output = backend.createTensorHandle(shape, dtype);

        // 根据不同的窗函数类型构建配置
        let args: WindowFuncKernelArgs;

        switch (kernelName) {
            case 'windowfunc.hann': {
                const config: GeneralizedCosineConfig = {
                    template: 'generalized_cosine',
                    coeffs: [0.5, 0.5, 0, 0],
                };
                args = { output, windowLength, denominator, config };
                break;
            }

            case 'windowfunc.hamming': {
                const alpha = (allParams['alpha'] ?? 0.54) as number;
                const beta = (allParams['beta'] ?? 0.46) as number;
                const config: GeneralizedCosineConfig = {
                    template: 'generalized_cosine',
                    coeffs: [alpha, beta, 0, 0],
                };
                args = { output, windowLength, denominator, config };
                break;
            }

            case 'windowfunc.blackman': {
                const config: GeneralizedCosineConfig = {
                    template: 'generalized_cosine',
                    coeffs: [0.42, 0.5, 0.08, 0],
                };
                args = { output, windowLength, denominator, config };
                break;
            }

            case 'windowfunc.bartlett': {
                const config: LinearConfig = {
                    template: 'linear',
                    windowType: 'bartlett',
                };
                args = { output, windowLength, denominator, config };
                break;
            }

            case 'windowfunc.kaiser': {
                const beta = (allParams['beta'] ?? 12.0) as number;
                const config: KaiserConfig = {
                    template: 'kaiser',
                    beta,
                };
                args = { output, windowLength, denominator, config };
                break;
            }

            default:
                throw new Error(`Unknown window function: ${kernelName}`);
        }

        // 获取 windowfunc kernel
        const kernel = backend.operators.find('windowfunc');
        if (!kernel) {
            throw new Error('WindowFunc kernel not available on backend');
        }

        // 构建 DirectContext 并调用 kernel
        const windowCtx: DirectContext = {
            kind: 'direct',
            inputs: [],
            scalars: args as any,
            metadata: {},
            outs: [output],
            kernelName: 'windowfunc',
        };

        (kernel as (ctx: DirectContext) => void)(windowCtx);

        return output;
    }

    static dispatch(entry: OpEntry, ctx: OperatorContext): ITensorHandle {
        const handler = WindowFuncHandler.getInstance();
        const execCtx = handler.buildContext(entry, ctx);
        return handler.execute(execCtx);
    }
}

export const dispatchWindowFunc = WindowFuncHandler.dispatch;
