/**
 * NN-Kit Operator Schema v7 - Ops Index
 *
 * 导出所有操作符定义
 * 按语义领域组织
 *
 * @module v7/ops
 */

// 逐元素运算 (合并 unary + arithmetic + comparison)
export * from './pointwise';

// 激活函数
export * from './activation';

// 归约运算
export * from './reduction';

// 扫描运算
export * from './scan';

// 线性代数
export * from './linalg';

// 三角矩阵操作
export * from './triangular';

// 归一化
export * from './norm';

// 卷积与池化
export * from './conv';

// 形状操作
export * from './shape';

// 索引操作 (合并 gather + scatter + embedding)
export * from './indexing';

// 排序操作
export * from './sort';

// 创建操作
export * from './creation';

// 内存操作
export * from './memory';

// 注意力机制
export * from './attention';

// FFT 频谱分析
export * from './fft';

// 音频处理
export * from './audio';

