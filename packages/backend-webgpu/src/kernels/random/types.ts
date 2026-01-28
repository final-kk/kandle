/**
 * Random Kernel Types
 * 
 * Type definitions for random number generation kernels
 */

/**
 * Random operation types
 */
export type RandomOpType = 'rand' | 'randn' | 'randint';

/**
 * Uniform buffer structure for random operations
 * 
 * Memory Layout (48 bytes, 3 × 16-byte aligned):
 * - numel: u32               // 4 bytes - 输出元素数量
 * - output_offset: u32       // 4 bytes - 输出偏移
 * - _pad0: u32               // 4 bytes
 * - _pad1: u32               // 4 bytes
 * - key0: u32                // 4 bytes - Philox 密钥低位
 * - key1: u32                // 4 bytes - Philox 密钥高位
 * - base_offset: u32         // 4 bytes - 全局偏移
 * - _pad2: u32               // 4 bytes
 * - low: i32                 // 4 bytes - randint only
 * - high: i32                // 4 bytes - randint only
 * - _pad3: u32               // 4 bytes
 * - _pad4: u32               // 4 bytes
 * Total: 48 bytes
 */
export interface RandomUniforms {
    /** Total number of output elements */
    numel: number;
    /** Output buffer offset */
    outputOffset: number;
    /** Philox key low 32 bits */
    key0: number;
    /** Philox key high 32 bits */
    key1: number;
    /** Base offset for counter (to ensure different calls produce different sequences) */
    baseOffset: number;
    /** Lower bound for randint (inclusive) */
    low: number;
    /** Upper bound for randint (exclusive) */
    high: number;
}

/**
 * Random shader parameters for shader generation
 */
export interface RandomShaderParams {
    opType: RandomOpType;
    outputDtype: string;
}
