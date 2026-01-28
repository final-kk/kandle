/**
 * Triangular Operations Registry
 * 
 * 注册 triu, tril 操作
 */

import type { TriangularOpConfig } from './types';

/**
 * Triangular 操作注册表
 */
export const TRIANGULAR_OPS: Record<string, TriangularOpConfig> = {
    /**
     * triu: 上三角矩阵提取
     * 
     * 保留 row <= col + diagonal 的元素
     * diagonal=0: 主对角线及以上
     * diagonal>0: 主对角线上方第 k 条对角线及以上
     * diagonal<0: 主对角线下方第 |k| 条对角线及以上
     */
    'triu': {
        name: 'triu',
        isUpper: true,
    },

    /**
     * tril: 下三角矩阵提取
     * 
     * 保留 row >= col + diagonal 的元素
     */
    'tril': {
        name: 'tril',
        isUpper: false,
    },
};
