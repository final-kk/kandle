/**
 * Triangular Matrix Kernel Types
 * 
 * 三角矩阵操作 (工业级实现：原生支持 strided 输入)
 * - triu: 上三角矩阵提取 (row <= col + diagonal)
 * - tril: 下三角矩阵提取 (row >= col + diagonal)
 * 
 * 参考: PyTorch ATen/native/TensorShape.cpp
 */

/**
 * Triangular 操作配置
 */
export interface TriangularOpConfig {
    /** 操作名称 */
    readonly name: string;
    /** 是否是上三角 (triu=true, tril=false) */
    readonly isUpper: boolean;
}

/**
 * Triangular Shader 参数 (工业级)
 * 
 * 支持 strided 输入，输出总是连续的
 */
export interface TriangularShaderParams {
    /** 操作配置 */
    config: TriangularOpConfig;
    /** 输入形状 */
    inputShape: readonly number[];
    /** 输入步幅 (支持非连续) */
    inputStrides: readonly number[];
    /** 输入偏移 (element count, not bytes) */
    inputOffset: number;
    /** 对角线偏移 */
    diagonal: number;
    /** WGSL 计算类型 */
    wgslType: string;
    /** workgroup 大小 */
    workgroupSize: number;
}
