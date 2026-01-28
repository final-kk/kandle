/**
 * MatrixHandler (v6)
 *
 * 处理矩阵 Kernel 操作: matmul, mm, bmm, mv, dot, addmm, baddbmm, addmv, outer
 * 直接调用专用 kernel，不使用 TensorIterator
 * 
 * 注意: linear, diag, trace 现在由 CompositeHandler 处理 (mechanism: 'Composite')
 */

import type { ITensorHandle } from '@kandle/types';
import type { OpEntry } from '@kandle/types';
import { dispatchMatmul } from '../matmulOps';
import type { PatternHandler, OperatorContext, DirectContext } from './types';

export class MatrixHandler implements PatternHandler {
    private static instance: MatrixHandler;

    static getInstance(): MatrixHandler {
        if (!MatrixHandler.instance) {
            MatrixHandler.instance = new MatrixHandler();
        }
        return MatrixHandler.instance;
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
            case 'matmul':
            case 'mm':
            case 'bmm':
            case 'mv':
            case 'dot': {
                const transposeA = (allParams['transposeA'] ?? false) as boolean;
                const transposeB = (allParams['transposeB'] ?? false) as boolean;
                // dispatchMatmul 签名: (a, b, c?, alpha?, beta?, out?, transposeA?, transposeB?)
                return dispatchMatmul(
                    inputs[0], inputs[1],
                    undefined, // c
                    1.0,       // alpha
                    0.0,       // beta
                    out,
                    transposeA,
                    transposeB
                );
            }

            case 'addmm': {
                // addmm: beta * self + alpha * (mat1 @ mat2)
                const self = inputs[0];
                const mat1 = inputs[1];
                const mat2 = inputs[2];
                const alpha = (allParams['alpha'] ?? 1) as number;
                const beta = (allParams['beta'] ?? 1) as number;
                return dispatchMatmul(mat1, mat2, self, alpha, beta, out);
            }

            case 'baddbmm': {
                // baddbmm: beta * self + alpha * (batch1 @ batch2)
                const self = inputs[0];
                const batch1 = inputs[1];
                const batch2 = inputs[2];
                const alpha = (allParams['alpha'] ?? 1) as number;
                const beta = (allParams['beta'] ?? 1) as number;
                return dispatchMatmul(batch1, batch2, self, alpha, beta, out);
            }

            case 'addmv':
            case 'outer':
                throw new Error(`${kernelName}: implementation pending`);

            default:
                throw new Error(`Unknown matrix operation: ${kernelName}`);
        }
    }

    static dispatch(entry: OpEntry, ctx: OperatorContext): ITensorHandle {
        const handler = MatrixHandler.getInstance();
        const execCtx = handler.buildContext(entry, ctx);
        return handler.execute(execCtx);
    }
}

export const dispatchMatrix = MatrixHandler.dispatch;
