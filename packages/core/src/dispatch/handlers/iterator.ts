/**
 * IteratorHandler (v6)
 *
 * 处理所有使用 TensorIterator 的操作:
 * - Map (Pointwise)
 * - Reduce (Reduction)
 * - Scan (Prefix Sum)
 */

import type { ITensorHandle } from '@kandle/types';
import type { OpEntry, ITensorIterator } from '@kandle/types';
import { TensorIterator } from '../TensorIterator';
import { env } from '../../env';
import { deriveTypePromotionKindFromRule } from '@kandle/utils';
import type {
    PatternHandler,
    OperatorContext,
    IteratorContext,
} from './types';

/**
 * IteratorHandler - 处理 Iterator 机制的操作
 */
export class IteratorHandler implements PatternHandler {
    private static instance: IteratorHandler;

    static getInstance(): IteratorHandler {
        if (!IteratorHandler.instance) {
            IteratorHandler.instance = new IteratorHandler();
        }
        return IteratorHandler.instance;
    }

    buildContext(entry: OpEntry, ctx: OperatorContext): IteratorContext {
        if (!entry.iteratorType) {
            throw new Error(`Iterator operation ${entry.name} missing iteratorType`);
        }

        let iterator: TensorIterator;

        switch (entry.iteratorType) {
            case 'Map':
                iterator = this.buildMapIterator(entry, ctx);
                break;
            case 'Reduce':
                iterator = this.buildReduceIterator(entry, ctx);
                break;
            case 'Scan':
                iterator = this.buildScanIterator(entry, ctx);
                break;
            default:
                throw new Error(`Unknown iteratorType: ${entry.iteratorType}`);
        }

        // 统一附加 scalarArgs
        // 注意: Reduction/Scan 可能已经在 build 过程中处理了部分参数，但这里再次设置通常无害
        // 主要是为了 Map 操作传递 alpha 等
        // 统一附加 scalarArgs
        // 注意: Reduction/Scan 可能已经在 build 过程中处理了部分参数 (如 dim)，需合并而非覆盖
        const currentArgs = iterator.getScalarArgs();
        iterator.setScalarArgs({ ...currentArgs, ...ctx.scalarArgs });

        return {
            kind: 'iterator',
            iterator,
            kernelName: entry.dispatchKey,
        };
    }

    execute(execCtx: IteratorContext): ITensorHandle | ITensorHandle[] {
        const { iterator, kernelName } = execCtx;
        const device = iterator.output().tensorHandle!.device;
        const backend = env.getBackend(device);

        const kernel = backend.operators.find(kernelName) as ((iter: ITensorIterator) => void) | undefined;
        if (!kernel) {
            throw new Error(`Kernel '${kernelName}' not found for device '${device}'`);
        }

        kernel(iterator);

        if (iterator.numOutputs > 1) {
            const outputs: ITensorHandle[] = [];
            for (let i = 0; i < iterator.numOutputs; i++) {
                outputs.push(iterator.output(i).tensorHandle!);
            }
            return outputs;
        }

        return iterator.output().tensorHandle!;
    }

    static dispatch(entry: OpEntry, ctx: OperatorContext): ITensorHandle | ITensorHandle[] {
        const handler = IteratorHandler.getInstance();
        const execCtx = handler.buildContext(entry, ctx);
        return handler.execute(execCtx);
    }

    // ========================================================================
    // Private Builders
    // ========================================================================

    private buildMapIterator(entry: OpEntry, ctx: OperatorContext): TensorIterator {
        const { tensorInputs, outs } = ctx;
        const out = outs?.[0];
        const config = entry.iteratorConfig!;
        const typePromotionKind = deriveTypePromotionKindFromRule(entry.dtype);

        // 确定输入
        let inputs: ITensorHandle[];
        switch (config.factory) {
            case 'unary':
                inputs = [tensorInputs[0]];
                break;
            case 'binary':
                inputs = [tensorInputs[0], tensorInputs[1]];
                break;
            case 'ternary':
                inputs = [tensorInputs[0], tensorInputs[1], tensorInputs[2]];
                break;
            default:
                // 如果没有 factory 或不匹配，尝试直接使用 tensorInputs
                inputs = [...tensorInputs];
        }

        const disableDimensionCoalescing = ['triu', 'tril'].includes(entry.dispatchKey);

        return TensorIterator.build({
            inputs,
            outputs: [out],
            opName: entry.dispatchKey,
            typePromotionKind,
            disableDimensionCoalescing,
        });
    }

    private buildReduceIterator(entry: OpEntry, ctx: OperatorContext): TensorIterator {
        const { tensorInputs, metadata, scalarArgs, outs } = ctx;
        const out = outs?.[0];
        const extraOut = outs?.[1]; // For reduction with indices

        const dim = metadata['dim'] ?? scalarArgs['dim'];
        // Reduce operations usually treat keepdim as boolean
        const keepdim = (metadata['keepdim'] ?? scalarArgs['keepdim'] ?? false) as boolean;

        // 计算 axes
        let axes: number[];
        const inputRank = tensorInputs[0].shape.length;

        if (dim === undefined || dim === null) {
            axes = Array.from({ length: inputRank }, (_, i) => i);
        } else if (Array.isArray(dim)) {
            axes = dim.map(d => d < 0 ? inputRank + d : d);
        } else {
            const d = dim as number;
            axes = [d < 0 ? inputRank + d : d];
        }

        // 判断是否是带索引的归约 (根据返回签名或 check extraOut)
        // 简单起见，如果提供了 extraOut，那就是带索引的
        const isReductionWithIndices = !!extraOut || this.hasIndicesOutput(entry);

        if (isReductionWithIndices) {
            return TensorIterator.reductionWithIndicesOp(
                tensorInputs[0],
                entry.dispatchKey,
                axes,
                keepdim,
                out,
                extraOut
            );
        } else {
            return TensorIterator.reductionOp(
                tensorInputs[0],
                entry.dispatchKey,
                axes,
                keepdim,
                out
            );
        }
    }

    private buildScanIterator(entry: OpEntry, ctx: OperatorContext): TensorIterator {
        const { tensorInputs, metadata, scalarArgs, outs } = ctx;
        const out = outs?.[0];
        const extraOut = outs?.[1];

        const dim = (metadata['dim'] ?? scalarArgs['dim']) as number;

        // Scan 通常需要 dim 参数
        if (dim === undefined) {
            throw new Error(`Scan operation ${entry.name} requires 'dim'`);
        }

        const isScanWithIndices = !!extraOut || this.hasIndicesOutput(entry);
        // FIXME: Check if TensorIterator has scanOp / scanWithIndicesOp exposed
        // Assuming implementation based on step 82 summary

        // 由于 TensorIterator.ts 是 User Space，我们假设它存在 scanOp
        // 实际上 TensorIterator.scanOp 可能需要类型断言或更新

        if (isScanWithIndices) {
            return (TensorIterator as any).scanWithIndicesOp(
                tensorInputs[0],
                entry.dispatchKey,
                dim,
                out,
                extraOut
            );
        } else {
            return (TensorIterator as any).scanOp(
                tensorInputs[0],
                entry.dispatchKey,
                dim,
                out
            );
        }
    }

    private hasIndicesOutput(entry: OpEntry): boolean {
        const returns = entry.signature.returns;
        if (!returns) return false;
        if ('tuple' in returns) {
            return returns.tuple.some(r =>
                r.name === 'indices' ||
                (r.type.kind === 'Tensor' && 'dtype' in r.type && r.type.dtype === 'int64')
            );
        }
        return false;
    }
}

export const dispatchIterator = IteratorHandler.dispatch;
