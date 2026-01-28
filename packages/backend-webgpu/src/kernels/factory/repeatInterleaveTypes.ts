/**
 * RepeatInterleave Kernel Types
 * 
 * Type definitions for repeat_interleave operation
 */

import { DType } from "@kandle/types";

/**
 * Uniform buffer structure for repeat_interleave (scalar repeats)
 * 
 * Memory Layout (64 bytes):
 * - numel: u32              // 4 bytes - 输出元素总数
 * - repeats: u32            // 4 bytes - 每个元素的重复次数
 * - inputNumel: u32         // 4 bytes - 输入元素总数
 * - rank: u32               // 4 bytes - 张量维度
 * - dim: i32                // 4 bytes - 沿哪个维度重复（-1 表示展平）
 * - inputOffset: u32        // 4 bytes - 输入偏移
 * - outputOffset: u32       // 4 bytes - 输出偏移
 * - _pad: u32               // 4 bytes
 * - inputShape: vec4<u32>   // 16 bytes - 输入形状 (最多支持 4D)
 * - inputStrides: vec4<i32> // 16 bytes - 输入步长
 * Total: 64 bytes
 */
export interface RepeatInterleaveUniforms {
    /** Total number of output elements */
    numel: number;
    /** Number of repeats per element */
    repeats: number;
    /** Total number of input elements */
    inputNumel: number;
    /** Rank of the tensor */
    rank: number;
    /** Dimension to repeat along (-1 for flattened) */
    dim: number;
    /** Input buffer offset */
    inputOffset: number;
    /** Output buffer offset */
    outputOffset: number;
    /** Input shape (padded to 4) */
    inputShape: number[];
    /** Input strides (padded to 4) */
    inputStrides: number[];
}

/**
 * Shader parameters for repeat_interleave
 */
export interface RepeatInterleaveShaderParams {
    /** Data type */
    dtype: DType;
    /** Rank of input tensor */
    rank: number;
    /** Whether dim is specified (false = flatten first) */
    hasDim: boolean;
}
