/**
 * View/Memory Layout Utilities
 * 
 * 提供 tensor view 操作所需的辅助函数，包括：
 * - 连续性检测
 * - Shape 规范化
 * - View/Slice/Permute/Select 的参数计算
 */

import { Shape } from "@kandle/types";
import { computeNumel } from "./shape";

/**
 * 检查 tensor 是否是连续的 (row-major contiguous)
 * 
 * 连续条件：strides 严格等于 shape 的 row-major 步长
 * 即 strides[i] = product(shape[i+1:])
 */
export function isContiguousStrides(
    shape: readonly number[],
    strides: readonly number[]
): boolean {
    if (shape.length === 0) {
        // 标量永远连续
        return true;
    }

    if (shape.length !== strides.length) {
        return false;
    }

    // 从右向左检查 strides
    let expectedStride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
        // 跳过大小为 1 的维度 (它们的 stride 可以是任意值)
        if (shape[i] !== 1) {
            if (strides[i] !== expectedStride) {
                return false;
            }
        }
        expectedStride *= shape[i];
    }

    return true;
}

/**
 * 检查是否可以对给定 tensor 执行 view 到目标 shape
 * 
 * View 要求 tensor 必须是连续的，或者满足特定的 stride 条件
 */
export function canView(
    currentShape: readonly number[],
    currentStrides: readonly number[],
    currentOffset: number,
    newShape: readonly number[]
): boolean {
    // 基本条件：numel 必须相等
    const currentNumel = computeNumel(currentShape as Shape);
    const newNumel = computeNumel(newShape as Shape);

    if (currentNumel !== newNumel) {
        return false;
    }

    // Issue #6: View 只要求数据在内存中是连续的
    // Offset 不影响 view 能力（只影响起始位置）
    return isContiguousStrides(currentShape, currentStrides);
}

/**
 * 规范化 shape，处理 -1 维度推断
 * 
 * @param shape 可能包含 -1 的 shape
 * @param numel 目标 numel
 * @returns 完整的 shape
 */
export function normalizeShape(
    shape: readonly number[],
    numel: number
): number[] {
    const result = [...shape];
    let negativeIndex = -1;
    let knownProduct = 1;

    for (let i = 0; i < result.length; i++) {
        if (result[i] === -1) {
            if (negativeIndex !== -1) {
                throw new Error("Only one dimension can be -1 in shape");
            }
            negativeIndex = i;
        } else if (result[i] < 0) {
            throw new Error(`Invalid shape dimension: ${result[i]}`);
        } else {
            knownProduct *= result[i];
        }
    }

    if (negativeIndex !== -1) {
        if (numel % knownProduct !== 0) {
            throw new Error(`Cannot reshape tensor of numel ${numel} to shape [${shape.join(', ')}]`);
        }
        result[negativeIndex] = numel / knownProduct;
    }

    // 验证 numel
    const resultNumel = result.reduce((a, b) => a * b, 1);
    if (resultNumel !== numel) {
        throw new Error(`Shape [${result.join(', ')}] is invalid for tensor of numel ${numel}`);
    }

    return result;
}


/**
 * 计算 select 操作后的新参数
 * 
 * select(axis, index) 会降低一个维度
 * 例如：shape [2, 3, 4], select(1, 2) -> shape [2, 4]
 */
export function computeSelectParams(
    shape: readonly number[],
    strides: readonly number[],
    offset: number,
    axis: number,
    index: number
): { newShape: number[], newStrides: number[], newOffset: number } {
    // 验证 axis
    if (axis < 0 || axis >= shape.length) {
        throw new Error(`select: axis ${axis} out of bounds for tensor of ndim ${shape.length}`);
    }

    // 处理负数索引
    let normalizedIndex = index;
    if (normalizedIndex < 0) {
        normalizedIndex += shape[axis];
    }

    // 验证 index
    if (normalizedIndex < 0 || normalizedIndex >= shape[axis]) {
        throw new Error(`select: index ${index} out of bounds for dimension ${axis} of size ${shape[axis]}`);
    }

    // 计算新 offset
    const newOffset = offset + normalizedIndex * strides[axis];

    // 移除该维度
    const newShape = [...shape.slice(0, axis), ...shape.slice(axis + 1)];
    const newStrides = [...strides.slice(0, axis), ...strides.slice(axis + 1)];

    return { newShape, newStrides, newOffset };
}

/**
 * 计算 slice 操作后的新参数
 * 
 * slice 保持维度数量，但可能改变每个维度的大小
 * 
 * TODO: 高级索引 (advanced indexing) 需要单独实现
 * 当前只支持基本切片：连续范围 + step
 */
export function computeSliceParams(
    shape: readonly number[],
    strides: readonly number[],
    offset: number,
    starts: number[],
    ends: number[],
    steps?: number[]
): { newShape: number[], newStrides: number[], newOffset: number } {
    const ndim = shape.length;
    const newShape: number[] = [];
    const newStrides: number[] = [];
    let newOffset = offset;

    for (let i = 0; i < ndim; i++) {
        const dimSize = shape[i];
        const dimStride = strides[i];

        // 获取 start, end, step
        let start = starts[i] ?? 0;
        let end = ends[i] ?? dimSize;
        const step = steps?.[i] ?? 1;

        // 验证 step
        if (step === 0) {
            throw new Error(`slice: step cannot be zero`);
        }

        // 规范化索引 - 需要区分正负步长
        if (step > 0) {
            // 正步长
            if (start < 0) start += dimSize;
            if (end < 0) end += dimSize;

            // clamp 到有效范围
            start = Math.max(0, Math.min(dimSize, start));
            end = Math.max(0, Math.min(dimSize, end));

            // 计算新维度大小
            const newDimSize = Math.max(0, Math.ceil((end - start) / step));

            // 更新 offset: 移动到 start 位置
            newOffset += start * dimStride;

            newShape.push(newDimSize);
            newStrides.push(dimStride * step);
        } else {
            // 负步长
            // 对于负步长，从 parseSliceString 返回的值:
            // - start: 可能是 dimSize-1 (表示从最后开始) 或其他值
            // - end: 可能是 -1-dimSize (表示到最开头的前一个位置，即包含 index 0)

            // 规范化 start
            if (start < 0) start += dimSize;
            start = Math.max(0, Math.min(dimSize - 1, start));

            // 规范化 end
            // end = -1 表示 "到 index 0 之前"，即包含 index 0
            // end < -dimSize 表示 "到数组开头之前"
            let effectiveEnd: number;
            if (end <= -dimSize) {
                // 表示包含 index 0，即走到最开头
                effectiveEnd = -1;  // -1 表示 "before index 0"
            } else if (end < 0) {
                effectiveEnd = end + dimSize;
            } else {
                effectiveEnd = end;
            }

            // 计算新维度大小
            // 从 start 往回走到 effectiveEnd (不包含 effectiveEnd)
            let newDimSize: number;
            if (effectiveEnd === -1) {
                // 走到最开头 (包含 index 0)
                newDimSize = Math.ceil((start + 1) / (-step));
            } else {
                newDimSize = Math.max(0, Math.ceil((start - effectiveEnd) / (-step)));
            }

            // 更新 offset: 移动到 start 位置
            newOffset += start * dimStride;

            newShape.push(newDimSize);
            newStrides.push(dimStride * step);  // step 是负数，所以 stride 也变负
        }
    }

    return { newShape, newStrides, newOffset };
}

/**
 * 计算 permute 操作后的新参数
 * 
 * permute 只重排 shape 和 strides 的顺序
 */
export function computePermuteParams(
    shape: readonly number[],
    strides: readonly number[],
    axes: number[]
): { newShape: number[], newStrides: number[] } {
    const ndim = shape.length;

    // 验证 axes
    if (axes.length !== ndim) {
        throw new Error(`permute: number of axes (${axes.length}) must match tensor dimensions (${ndim})`);
    }

    // 验证 axes 是 0 到 ndim-1 的排列
    const seen = new Set<number>();
    for (const axis of axes) {
        if (axis < 0 || axis >= ndim) {
            throw new Error(`permute: axis ${axis} out of bounds for tensor of ndim ${ndim}`);
        }
        if (seen.has(axis)) {
            throw new Error(`permute: repeated axis ${axis}`);
        }
        seen.add(axis);
    }

    // 重排
    const newShape = axes.map(i => shape[i]);
    const newStrides = axes.map(i => strides[i]);

    return { newShape, newStrides };
}

/**
 * 检查 slice 参数是否需要高级索引
 * 
 * 高级索引的标志：
 * - indices 是数组（如 tensor[[1,3,5]]）
 * - 布尔索引
 * 
 * 当前不支持，需要抛出错误
 */
export function requiresAdvancedIndexing(
    starts: readonly (number | number[])[],
    ends: readonly (number | number[])[],
    steps?: readonly number[]
): boolean {
    // 检查是否有数组类型的索引
    for (const s of starts) {
        if (Array.isArray(s)) return true;
    }
    for (const e of ends) {
        if (Array.isArray(e)) return true;
    }

    return false;
}
