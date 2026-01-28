/**
 * IIR Filter Kernel Types
 * 
 * 定义 IIR 滤波器的状态空间表示和矩阵 scan 配置
 * 
 * 理论基础:
 * IIR 差分方程: y[n] = sum(b[i]*x[n-i]) - sum(a[j]*y[n-j])
 * 
 * 转换为状态空间形式:
 * s[n] = A * s[n-1] + B * x[n]
 * y[n] = C * s[n] + D * x[n]
 * 
 * 对于 biquad (二阶 IIR), 状态向量维度为 2
 * 使用扩展状态向量 [y[n], y[n-1], 1]^T 可将递推转为纯矩阵乘法:
 * s[n] = M[n] * s[n-1]
 * 
 * 由于矩阵乘法满足结合律, 可用 parallel prefix scan 并行计算
 */

/**
 * IIR 滤波器阶数类型
 * 当前只支持 biquad (二阶)
 */
export type IIROrder = 2;

/**
 * Biquad 滤波器系数
 * 对应传递函数 H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)
 */
export interface BiquadCoeffs {
    /** 分子系数 (归一化后, a0=1) */
    b0: number;
    b1: number;
    b2: number;
    /** 分母系数 (归一化后 a0=1, 只需 a1, a2) */
    a1: number;
    a2: number;
}

/**
 * IIR Scan 执行参数
 */
export interface IIRScanParams {
    /** 信号长度 (最后一维大小) */
    signalLength: number;

    /** Batch 大小 (除最后一维外所有维度的乘积) */
    batchSize: number;

    /** 滤波器系数 */
    coeffs: BiquadCoeffs;

    /** 是否限制输出范围 */
    clamp: boolean;
    clampMin: number;
    clampMax: number;
}

/**
 * Workgroup 配置
 */
export const IIR_WORKGROUP_SIZE = 256;

/**
 * 每个 block 处理的信号样本数
 * 必须是 workgroup size 的倍数
 */
export const IIR_ELEMENTS_PER_BLOCK = 512;

/**
 * 状态矩阵维度 (biquad: 3x3)
 * 扩展状态向量: [y[n], y[n-1], 1]
 */
export const IIR_STATE_DIM = 3;
