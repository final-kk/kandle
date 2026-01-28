/**
 * v6 TriangularHandler
 *
 * 处理三角矩阵操作: triu, tril
 * 透传 DirectContext 给后端 kernel
 */

import type { ITensorHandle } from '@kandle/types';
import type { OpEntry } from '@kandle/types';
import { env } from '../../env';
import type { PatternHandler, OperatorContext, DirectContext } from './types';

export class TriangularHandler implements PatternHandler {
    private static instance: TriangularHandler;

    static getInstance(): TriangularHandler {
        if (!TriangularHandler.instance) {
            TriangularHandler.instance = new TriangularHandler();
        }
        return TriangularHandler.instance;
    }

    buildContext(entry: OpEntry, ctx: OperatorContext): DirectContext {
        const input = ctx.tensorInputs[0];
        const backend = env.getBackend(input.device);

        // 预分配输出张量
        const output = backend.createTensorHandle(input.shape, input.dtype);

        return {
            kind: 'direct',
            inputs: ctx.tensorInputs,
            scalars: ctx.scalarArgs,
            metadata: ctx.metadata,
            outs: [output],
            kernelName: entry.dispatchKey,
        };
    }

    execute(execCtx: DirectContext): ITensorHandle {
        const { inputs, kernelName } = execCtx;
        const backend = env.getBackend(inputs[0].device);

        // 获取 kernel 并直接透传 DirectContext
        const kernel = backend.operators.find(kernelName);
        if (!kernel) {
            throw new Error(`Triangular kernel '${kernelName}' not found`);
        }

        // 直接透传 DirectContext
        (kernel as (ctx: DirectContext) => void)(execCtx);

        return execCtx.outs![0];
    }

    static dispatch(entry: OpEntry, ctx: OperatorContext): ITensorHandle {
        const handler = TriangularHandler.getInstance();
        const execCtx = handler.buildContext(entry, ctx);
        return handler.execute(execCtx);
    }
}

export const dispatchTriangular = TriangularHandler.dispatch;
