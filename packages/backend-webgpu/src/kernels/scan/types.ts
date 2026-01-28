/**
 * Scan Kernel Types
 * 
 * 定义扫描操作 (Prefix Sum) 的配置接口
 * 参考: GPU Gems 3 Chapter 39 - Parallel Prefix Sum (Scan)
 * 参考: PyTorch cumsum/cumprod 实现
 */

/**
 * 扫描操作配置
 * 每个 dispatchKey 对应一个配置
 */
export interface ScanOpConfig {
    /**
     * 二元结合操作符
     * @param a - 左操作数变量名
     * @param b - 右操作数变量名
     * @param computeType - WGSL 计算类型 'f32' | 'i32' | 'u32'
     * @returns WGSL 表达式
     */
    operator: (a: string, b: string, computeType: string) => string;

    /**
     * 恒等元 (Identity Element)
     * 对于 sum: 0, 对于 prod: 1, 对于 max: -INF, 对于 min: +INF
     * @param computeType - WGSL 计算类型
     * @returns WGSL 表达式
     */
    identity: (computeType: string) => string;

    /**
     * 是否返回 indices (用于 cummax/cummin)
     */
    hasIndices: boolean;

    /**
     * 比较操作符 (仅 cummax/cummin)
     * 返回 true 如果 a 应该替换之前的值
     * @param newVal - 新值变量名
     * @param curVal - 当前值变量名
     * @returns WGSL bool 表达式
     */
    compare?: (newVal: string, curVal: string) => string;
}

/**
 * 扫描维度参数
 * 用于在 executor 中解析扫描维度信息
 */
export interface ScanDimParams {
    /** 扫描维度索引 (已规范化) */
    scanDim: number;

    /** 扫描维度的大小 */
    scanDimSize: number;

    /** 外层大小: product of dims before scanDim */
    outerSize: number;

    /** 内层大小: product of dims after scanDim */
    innerSize: number;

    /** 输入张量形状 */
    inputShape: readonly number[];

    /** 输入张量步幅 */
    inputStrides: readonly number[];
}

/**
 * 扫描策略
 */
export type ScanStrategy = 'single-pass' | 'multi-pass';

/**
 * 扫描策略阈值
 * 当 scanDimSize <= 此值时使用单 pass，否则使用多 pass
 */
export const SCAN_SINGLE_PASS_THRESHOLD = 1024;

/**
 * 扫描共享内存大小 (elements)
 * 对于 Blelloch 算法，需要 2 * workgroupSize 空间
 * 加上 bank conflict padding
 */
export const SCAN_SHARED_MEM_SIZE = (workgroupSize: number) =>
    workgroupSize * 2 + Math.ceil((workgroupSize * 2) / 16);
