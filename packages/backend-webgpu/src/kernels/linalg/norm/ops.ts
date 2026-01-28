/**
 * Norm Operations Registry
 * 
 * 注册 norm 操作
 */

import type { NormOpConfig } from './types';

/**
 * Norm 操作注册表
 */
export const NORM_OPS: Record<string, NormOpConfig> = {
    /**
     * norm: Lp 范数
     * 
     * 公式: (sum(|x|^p))^(1/p)
     * 
     * 参考: PyTorch torch.linalg.vector_norm
     */
    'norm': {
        name: 'norm',
    },
};
