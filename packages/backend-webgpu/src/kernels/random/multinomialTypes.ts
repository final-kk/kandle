/**
 * Multinomial Kernel Types
 * 
 * Type definitions for multinomial sampling kernels
 */

/**
 * Uniform buffer structure for multinomial operation
 * 
 * Memory Layout (80 bytes):
 * - batchSize: u32          // 4 bytes - batch维度大小（1D时为1）
 * - numClasses: u32         // 4 bytes - 类别数量
 * - numSamples: u32         // 4 bytes - 采样数量
 * - replacement: u32        // 4 bytes - 是否有放回采样 (0 or 1)
 * - inputOffset: u32        // 4 bytes - 输入缓冲区偏移
 * - outputOffset: u32       // 4 bytes - 输出缓冲区偏移
 * - _pad0: u32              // 4 bytes
 * - _pad1: u32              // 4 bytes
 * - key0: u32               // 4 bytes - Philox 密钥低位
 * - key1: u32               // 4 bytes - Philox 密钥高位
 * - baseOffset: u32         // 4 bytes - 全局偏移
 * - _pad2: u32              // 4 bytes
 * Total: 48 bytes
 */
export interface MultinomialUniforms {
    /** Batch size (1 for 1D input) */
    batchSize: number;
    /** Number of classes (size of last dimension) */
    numClasses: number;
    /** Number of samples to draw */
    numSamples: number;
    /** Whether to sample with replacement (0 or 1) */
    replacement: number;
    /** Input buffer offset */
    inputOffset: number;
    /** Output buffer offset */
    outputOffset: number;
    /** Philox key low 32 bits */
    key0: number;
    /** Philox key high 32 bits */
    key1: number;
    /** Base offset for counter */
    baseOffset: number;
}

/**
 * Input strides for multinomial (used for non-contiguous tensors)
 */
export interface MultinomialInputMeta {
    /** Strides for input tensor */
    strides: number[];
    /** Shape of input tensor */
    shape: number[];
    /** Offset of input tensor */
    offset: number;
}

/**
 * Multinomial shader parameters for shader generation
 */
export interface MultinomialShaderParams {
    /** Number of classes the shader is configured for */
    numClasses: number;
    /** Whether to sample with replacement */
    replacement: boolean;
}
