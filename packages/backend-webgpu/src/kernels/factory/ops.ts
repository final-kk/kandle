/**
 * Factory Operations Registry
 * 
 * 注册 eye 等工厂操作
 */

import type { EyeOpConfig } from './types';

/**
 * Eye 操作注册表
 */
export const EYE_OPS: Record<string, EyeOpConfig> = {
    /**
     * eye: 单位矩阵
     * 
     * 创建 n x m 矩阵，对角线为 1，其余为 0
     */
    'eye': {
        name: 'eye',
    },
};
