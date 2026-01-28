/**
 * MemoryFormat Utilities
 * 
 * 支持 NCHW (Contiguous) 和 NHWC (ChannelsLast) 内存布局格式。
 * 设计对标 PyTorch 的 memory_format 实现。
 * 
 * 核心概念:
 * - 逻辑形状 (shape) 保持不变，物理布局通过 strides 表达
 * - ChannelsLast 将 channel 维度变为最"密集"的维度 (stride=1)
 * 
 * @module utils/memoryFormat
 */

import { MemoryFormat, Shape } from "@kandle/types";

/**
 * 根据 MemoryFormat 计算 strides
 * 
 * @param shape - 张量形状 (逻辑形状，如 [N, C, H, W])
 * @param format - 目标内存格式
 * @returns strides 数组
 * 
 * @example
 * // Contiguous (NCHW): shape [2, 3, 4, 5] → strides [60, 20, 5, 1]
 * computeStridesForFormat([2, 3, 4, 5], MemoryFormat.Contiguous);
 * 
 * // ChannelsLast (NHWC): shape [2, 3, 4, 5] → strides [60, 1, 15, 3]
 * computeStridesForFormat([2, 3, 4, 5], MemoryFormat.ChannelsLast);
 */
export function computeStridesForFormat(shape: Shape, format: MemoryFormat): number[] {
    const ndim = shape.length;

    // 标量或 0 维张量
    if (ndim === 0) {
        return [];
    }

    // 1-3 维张量始终使用 Contiguous
    if (ndim < 4) {
        return computeContiguousStrides(shape);
    }

    switch (format) {
        case MemoryFormat.Contiguous:
        case MemoryFormat.Preserve:
            return computeContiguousStrides(shape);

        case MemoryFormat.ChannelsLast:
            if (ndim !== 4) {
                throw new Error(
                    `ChannelsLast format requires 4D tensor, got ${ndim}D. ` +
                    `Use ChannelsLast3d for 5D tensors.`
                );
            }
            return computeChannelsLastStrides4D(shape);

        case MemoryFormat.ChannelsLast3d:
            if (ndim !== 5) {
                throw new Error(
                    `ChannelsLast3d format requires 5D tensor, got ${ndim}D. ` +
                    `Use ChannelsLast for 4D tensors.`
                );
            }
            return computeChannelsLastStrides5D(shape);

        default:
            throw new Error(`Unsupported MemoryFormat: ${format}`);
    }
}

/**
 * 标准 row-major (C-contiguous) strides 计算
 * 
 * 从右向左累乘: shape [N, C, H, W] → strides [C*H*W, H*W, W, 1]
 */
function computeContiguousStrides(shape: Shape): number[] {
    const ndim = shape.length;
    const strides = new Array(ndim);
    let stride = 1;
    for (let i = ndim - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

/**
 * ChannelsLast (NHWC) strides 计算 - 4D 专用
 * 
 * 逻辑形状: [N, C, H, W]
 * 物理排列: N → H → W → C
 * Strides: [H*W*C, 1, W*C, C]
 * 
 * 关键: channel 维度的 stride = 1 (最密集)
 */
function computeChannelsLastStrides4D(shape: Shape): number[] {
    const [N, C, H, W] = shape;
    // 物理布局: NHWC
    // strides 顺序对应逻辑形状 [N, C, H, W]
    return [
        H * W * C,  // N: 跳过整个 H*W*C 块
        1,          // C: 最密集，相邻 channel 相邻存储
        W * C,      // H: 跳过一行 (W*C 元素)
        C           // W: 跳过一个像素 (C 个 channel)
    ];
}

/**
 * ChannelsLast3d (NDHWC) strides 计算 - 5D 专用
 * 
 * 逻辑形状: [N, C, D, H, W]
 * 物理排列: N → D → H → W → C
 * Strides: [D*H*W*C, 1, H*W*C, W*C, C]
 */
function computeChannelsLastStrides5D(shape: Shape): number[] {
    const [N, C, D, H, W] = shape;
    return [
        D * H * W * C,  // N
        1,              // C: 最密集
        H * W * C,      // D
        W * C,          // H
        C               // W
    ];
}

/**
 * 从 strides 推断 MemoryFormat
 * 
 * 用于从现有张量检测其内存格式，以便在操作中保持格式一致性。
 * 
 * @param shape - 张量形状
 * @param strides - 张量 strides
 * @returns 推断出的 MemoryFormat
 * 
 * @example
 * inferMemoryFormat([2, 3, 4, 5], [60, 1, 15, 3]); // ChannelsLast
 * inferMemoryFormat([2, 3, 4, 5], [60, 20, 5, 1]); // Contiguous
 */
export function inferMemoryFormat(shape: Shape, strides: readonly number[]): MemoryFormat {
    const ndim = shape.length;

    // 低于 4 维始终视为 Contiguous
    if (ndim < 4) {
        return MemoryFormat.Contiguous;
    }

    // 4D: 检查 ChannelsLast
    if (ndim === 4) {
        const expectedCL = computeChannelsLastStrides4D(shape);
        if (stridesEqual(strides, expectedCL)) {
            return MemoryFormat.ChannelsLast;
        }
    }

    // 5D: 检查 ChannelsLast3d
    if (ndim === 5) {
        const expectedCL3d = computeChannelsLastStrides5D(shape);
        if (stridesEqual(strides, expectedCL3d)) {
            return MemoryFormat.ChannelsLast3d;
        }
    }

    // 默认 Contiguous
    return MemoryFormat.Contiguous;
}

/**
 * 检查 strides 是否相等
 */
function stridesEqual(a: readonly number[], b: readonly number[]): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) return false;
    }
    return true;
}

/**
 * 检查张量是否为 ChannelsLast 格式
 * 
 * 包括 ChannelsLast (4D) 和 ChannelsLast3d (5D)
 */
export function isChannelsLast(format: MemoryFormat): boolean {
    return format === MemoryFormat.ChannelsLast || format === MemoryFormat.ChannelsLast3d;
}

/**
 * 为给定的 ndim 选择合适的 ChannelsLast 格式
 */
export function getChannelsLastFormat(ndim: number): MemoryFormat {
    if (ndim === 4) return MemoryFormat.ChannelsLast;
    if (ndim === 5) return MemoryFormat.ChannelsLast3d;
    throw new Error(`ChannelsLast format only supports 4D or 5D tensors, got ${ndim}D`);
}

/**
 * 检查 strides 是否为标准连续格式 (Contiguous)
 * 
 * 用于优化：连续张量可以使用更高效的 kernel 路径
 */
export function isContiguousFormat(shape: Shape, strides: readonly number[]): boolean {
    const expectedContiguous = computeContiguousStrides(shape);
    return stridesEqual(strides, expectedContiguous);
}
