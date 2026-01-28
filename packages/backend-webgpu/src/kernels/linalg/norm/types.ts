/**
 * Lp Norm Kernel Types
 * 
 * Lp 范数计算
 * 公式: (sum(|x|^p))^(1/p)
 * 
 * 特殊情况:
 * - p=0: 非零元素计数 (伪范数)
 * - p=1: L1 范数 sum(|x|)
 * - p=2: L2 范数 sqrt(sum(x²)) (欧几里得范数)
 * - p=inf: max(|x|)
 * - p=-inf: min(|x|)
 * 
 * 参考: PyTorch torch.linalg.vector_norm
 */

/**
 * Norm 操作配置
 */
export interface NormOpConfig {
    /** 操作名称 */
    readonly name: string;
}

/**
 * Norm 的 p 值类型
 */
export type NormOrd = number | 'inf' | '-inf';

/**
 * 标准化 p 值到数值（用于 uniform buffer）
 * inf -> 1e38, -inf -> -1e38
 */
export function normalizeOrd(p: NormOrd): number {
    if (p === 'inf' || p === Infinity) return 1e38;
    if (p === '-inf' || p === -Infinity) return -1e38;
    return p;
}

/**
 * 判断 p 值类型
 */
export function getNormType(p: NormOrd): 'zero' | 'one' | 'two' | 'inf' | 'neg_inf' | 'general' {
    if (p === 0) return 'zero';
    if (p === 1) return 'one';
    if (p === 2) return 'two';
    if (p === 'inf' || p === Infinity) return 'inf';
    if (p === '-inf' || p === -Infinity) return 'neg_inf';
    return 'general';
}
