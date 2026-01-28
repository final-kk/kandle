/**
 * v5 Pattern Handler Types
 *
 * 类型定义已迁移到 @kandle/types/dispatch
 * 此文件重新导出类型并提供推导工具函数
 */

import type { DType } from '@kandle/types';
import type { OpEntry } from '@kandle/types';

// Re-export types from @kandle/types
export type {
    OperatorContext,
    ExecutionContext,
    IteratorContext,
    DirectContext,
    MetadataContext,
    PatternHandler,
    DirectKernelImpl,
} from '@kandle/types';

// Import for use in this file
import type { OperatorContext } from '@kandle/types';

// ============================================================================
// Shape/DType Inference Utilities
// ============================================================================

/**
 * 根据 ShapeRule 推导输出形状
 */
export function inferShape(
    rule: OpEntry['shape'],
    ctx: OperatorContext
): number[] {
    const { tensorInputs, metadata } = ctx;

    switch (rule.rule) {
        case 'same': {
            // 找到对应的 tensor
            const asName = rule.as;
            // 简化: 假设第一个 tensor 是 'self'
            return [...tensorInputs[0].shape];
        }

        case 'broadcast': {
            // 广播多个 tensor
            let result = [...tensorInputs[0].shape];
            for (let i = 1; i < tensorInputs.length; i++) {
                result = broadcastShapes(result, tensorInputs[i].shape);
            }
            return result;
        }

        case 'reduction': {
            const input = tensorInputs[0];
            const axis = metadata[rule.axis] as number | number[] | undefined;
            const keepdims = (metadata[rule.keepdims] ?? false) as boolean;
            return computeReductionShape(input.shape, axis, keepdims);
        }

        case 'explicit': {
            // 从 metadata 或表达式获取
            const expr = rule.expr;
            if (expr === '[]') return []; // scalar output
            if (metadata['shape']) return metadata['shape'] as number[];
            if (metadata['size']) return metadata['size'] as number[];
            return [];
        }

        case 'matmul': {
            // 矩阵乘法形状推导
            const left = tensorInputs[0];
            const right = tensorInputs[1];
            return computeMatmulShape(left.shape, right.shape);
        }

        case 'permute': {
            const input = tensorInputs[0];
            const dims = metadata[rule.dims] as number[];
            return dims.map(d => input.shape[d < 0 ? input.shape.length + d : d]);
        }

        case 'transpose': {
            const input = tensorInputs[0];
            const dim0 = metadata[rule.dim0] as number;
            const dim1 = metadata[rule.dim1] as number;
            const shape = [...input.shape];
            const ndim = shape.length;
            const d0 = dim0 < 0 ? ndim + dim0 : dim0;
            const d1 = dim1 < 0 ? ndim + dim1 : dim1;
            [shape[d0], shape[d1]] = [shape[d1], shape[d0]];
            return shape;
        }

        default:
            throw new Error(`Unknown shape rule: ${(rule as any).rule}`);
    }
}

/**
 * 根据 DTypeRule 推导输出类型
 */
export function inferDtype(
    rule: OpEntry['dtype'],
    ctx: OperatorContext
): DType {
    const { tensorInputs, metadata } = ctx;

    switch (rule.rule) {
        case 'same':
            return tensorInputs[0].dtype;

        case 'promote':
            // 简化: 返回第一个 tensor 的类型
            // TODO: 实现真正的类型提升
            return tensorInputs[0].dtype;

        case 'fixed':
            return rule.dtype as DType;

        case 'explicit': {
            const dt = metadata[rule.param] as DType | undefined;
            if (dt) return dt;
            if (rule.fallback) {
                // fallback 可能是另一个 tensor 的名字
                return tensorInputs[0].dtype;
            }
            return tensorInputs[0].dtype;
        }

        default:
            throw new Error(`Unknown dtype rule: ${(rule as any).rule}`);
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

function broadcastShapes(a: readonly number[], b: readonly number[]): number[] {
    const result: number[] = [];
    const maxLen = Math.max(a.length, b.length);

    for (let i = 0; i < maxLen; i++) {
        const dimA = a[a.length - 1 - i] ?? 1;
        const dimB = b[b.length - 1 - i] ?? 1;

        if (dimA === dimB) {
            result.unshift(dimA);
        } else if (dimA === 1) {
            result.unshift(dimB);
        } else if (dimB === 1) {
            result.unshift(dimA);
        } else {
            throw new Error(`Cannot broadcast shapes [${a}] and [${b}]`);
        }
    }

    return result;
}

function computeReductionShape(
    shape: readonly number[],
    axis: number | number[] | undefined,
    keepdims: boolean
): number[] {
    if (axis === undefined) {
        // 全局归约
        return keepdims ? shape.map(() => 1) : [];
    }

    const axes = Array.isArray(axis) ? axis : [axis];
    const normalizedAxes = axes.map(a => (a < 0 ? shape.length + a : a));

    if (keepdims) {
        return shape.map((dim, i) => (normalizedAxes.includes(i) ? 1 : dim));
    } else {
        return shape.filter((_, i) => !normalizedAxes.includes(i));
    }
}

function computeMatmulShape(a: readonly number[], b: readonly number[]): number[] {
    // 1D @ 1D -> scalar
    if (a.length === 1 && b.length === 1) {
        return [];
    }

    // 1D @ 2D -> [N]
    if (a.length === 1 && b.length === 2) {
        return [b[1]];
    }

    // 2D @ 1D -> [M]
    if (a.length === 2 && b.length === 1) {
        return [a[0]];
    }

    // 2D @ 2D -> [M, N]
    if (a.length === 2 && b.length === 2) {
        return [a[0], b[1]];
    }

    // Batched: broadcast batch dims + [M, N]
    const aBatch = a.slice(0, -2);
    const bBatch = b.slice(0, -2);
    const batchShape = broadcastShapes(aBatch, bBatch);

    const M = a[a.length - 2];
    const N = b[b.length - 1];

    return [...batchShape, M, N];
}
