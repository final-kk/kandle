/**
 * FlashAttention Types
 * 
 * 定义 FlashAttention kernel 的配置类型和 uniform layouts
 */

import type { DType } from '@kandle/types';

// ============================================================================
// FlashAttention Configuration
// ============================================================================

/**
 * FlashAttention 执行配置
 */
export interface FlashAttentionConfig {
    /** Batch size */
    batchSize: number;
    /** Number of query heads */
    numHeadsQ: number;
    /** Number of KV heads (for GQA support) */
    numHeadsKV: number;
    /** Query sequence length (L) */
    seqLenQ: number;
    /** Key/Value sequence length (S) */
    seqLenKV: number;
    /** Head dimension / embedding dimension per head */
    headDim: number;
    /** Scaling factor (default: 1/sqrt(headDim)) */
    scale: number;
    /** Whether to apply causal masking */
    isCausal: boolean;
    /** Compute dtype */
    dtype: DType;
}

/**
 * Tile 配置 - 控制 tiling 策略
 */
export interface FlashAttentionTileConfig {
    /** Block size for Q dimension (Br) */
    blockSizeQ: number;
    /** Block size for KV dimension (Bc) */
    blockSizeKV: number;
    /** Workgroup size X */
    workgroupSizeX: number;
    /** Workgroup size Y */
    workgroupSizeY: number;
}

/**
 * 选择 tile 配置
 * 
 * 根据 head dimension 和序列长度选择最优的 block size
 * 
 * WebGPU workgroup memory 限制为 16KB (默认)
 * 
 * 需要存储 (in float32, 4 bytes each):
 * - Q_shared: blockSizeQ * headDim
 * - K_shared: blockSizeKV * headDim
 * - V_shared: blockSizeKV * headDim
 * - S_shared: blockSizeQ * blockSizeKV
 * - O_shared: blockSizeQ * headDim
 * - m_shared: blockSizeQ
 * - l_shared: blockSizeQ
 * 
 * Total = (2*Br*d + 2*Bc*d + Br*Bc + 2*Br) * 4 bytes
 */
export function selectTileConfig(headDim: number, seqLenQ: number, seqLenKV: number): FlashAttentionTileConfig {
    const MAX_WORKGROUP_STORAGE = 16384; // 16KB
    const BYTES_PER_ELEMENT = 4; // float32

    // 计算给定 block sizes 所需的 workgroup storage
    function calcStorage(br: number, bc: number): number {
        // Q: br * d, K: bc * d, V: bc * d, S: br * bc, O: br * d, m: br, l: br
        return (br * headDim + bc * headDim + bc * headDim + br * bc + br * headDim + br + br) * BYTES_PER_ELEMENT;
    }

    // 从最大 block size 开始，逐步减小直到满足限制
    let blockSizeQ = 32;
    let blockSizeKV = 32;

    // 尝试不同的 block sizes
    const candidates = [32, 16, 8, 4];

    for (const bq of candidates) {
        for (const bkv of candidates) {
            const storage = calcStorage(bq, bkv);
            if (storage <= MAX_WORKGROUP_STORAGE) {
                blockSizeQ = bq;
                blockSizeKV = bkv;
                // Found valid config, break both loops
                break;
            }
        }
        // Check if we found valid config
        if (calcStorage(blockSizeQ, blockSizeKV) <= MAX_WORKGROUP_STORAGE) {
            break;
        }
    }

    // 确保 block size 不超过序列长度
    blockSizeQ = Math.min(blockSizeQ, seqLenQ);
    blockSizeKV = Math.min(blockSizeKV, seqLenKV);

    // Fallback: 如果仍然超限，使用最小配置
    if (calcStorage(blockSizeQ, blockSizeKV) > MAX_WORKGROUP_STORAGE) {
        blockSizeQ = 4;
        blockSizeKV = 4;
    }

    // Workgroup size: 每个线程处理 block 内的一部分
    const workgroupSizeX = Math.min(blockSizeQ, 16);
    const workgroupSizeY = Math.min(headDim, 16);

    return {
        blockSizeQ,
        blockSizeKV,
        workgroupSizeX,
        workgroupSizeY,
    };
}

// ============================================================================
// Uniform Buffer Layout
// ============================================================================

/**
 * FlashAttention Uniform Buffer Layout
 * 
 * 按 16-byte 对齐
 */
export const FLASH_ATTENTION_UNIFORM_LAYOUT = {
    size: 64, // 16 * 4 bytes
    fields: {
        // vec4<u32> [0]: dimensions
        batchSize: 0,      // offset 0
        numHeadsQ: 4,      // offset 4
        numHeadsKV: 8,     // offset 8
        headDim: 12,       // offset 12

        // vec4<u32> [1]: sequence lengths
        seqLenQ: 16,       // offset 16
        seqLenKV: 20,      // offset 20
        blockSizeQ: 24,    // offset 24
        blockSizeKV: 28,   // offset 28

        // vec4<f32> [2]: scale and offsets
        scale: 32,         // offset 32 (f32)
        offsetQ: 36,       // offset 36 (u32)
        offsetK: 40,       // offset 40 (u32)
        offsetV: 44,       // offset 44 (u32)

        // vec4<u32> [3]: output offset and flags
        offsetO: 48,       // offset 48 (u32)
        isCausal: 52,      // offset 52 (u32, 0 or 1)
        _padding1: 56,     // offset 56
        _padding2: 60,     // offset 60
    },
};

/**
 * 创建 FlashAttention Uniform Buffer 数据
 */
export function createFlashAttentionUniformBuffer(params: {
    batchSize: number;
    numHeadsQ: number;
    numHeadsKV: number;
    headDim: number;
    seqLenQ: number;
    seqLenKV: number;
    blockSizeQ: number;
    blockSizeKV: number;
    scale: number;
    offsetQ: number;
    offsetK: number;
    offsetV: number;
    offsetO: number;
    isCausal: boolean;
}): ArrayBuffer {
    const buffer = new ArrayBuffer(FLASH_ATTENTION_UNIFORM_LAYOUT.size);
    const u32View = new Uint32Array(buffer);
    const f32View = new Float32Array(buffer);

    // vec4<u32> [0]: dimensions
    u32View[0] = params.batchSize;
    u32View[1] = params.numHeadsQ;
    u32View[2] = params.numHeadsKV;
    u32View[3] = params.headDim;

    // vec4<u32> [1]: sequence lengths
    u32View[4] = params.seqLenQ;
    u32View[5] = params.seqLenKV;
    u32View[6] = params.blockSizeQ;
    u32View[7] = params.blockSizeKV;

    // vec4<f32/u32> [2]: scale and offsets
    f32View[8] = params.scale;
    u32View[9] = params.offsetQ;
    u32View[10] = params.offsetK;
    u32View[11] = params.offsetV;

    // vec4<u32> [3]: output offset and flags
    u32View[12] = params.offsetO;
    u32View[13] = params.isCausal ? 1 : 0;
    u32View[14] = 0; // padding
    u32View[15] = 0; // padding

    return buffer;
}

// ============================================================================
// Cache Key Generation
// ============================================================================

/**
 * 生成 pipeline cache key
 */
export function computeFlashAttentionCacheKey(
    dtype: DType,
    headDim: number,
    blockSizeQ: number,
    blockSizeKV: number,
    isCausal: boolean,
    numHeadsQ: number,
    numHeadsKV: number,
): string {
    const gqaRatio = numHeadsQ / numHeadsKV;
    return `flash-attn:${dtype}:d${headDim}:bq${blockSizeQ}:bkv${blockSizeKV}:causal${isCausal ? 1 : 0}:gqa${gqaRatio}`;
}
