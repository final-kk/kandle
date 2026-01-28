/**
 * FlashAttention Kernel Module
 * 
 * 导出 FlashAttention kernel 高层 API
 */

import type { IBackendOpsRegister } from '@kandle/types';
import { flashAttention, executeFlashAttention } from './executor';

export { type FlashAttentionConfig, type FlashAttentionTileConfig, selectTileConfig } from './types';
export { flashAttention, executeFlashAttention };

/**
 * 注册 Attention kernels
 */
export function registerAttentionKernels(registry: IBackendOpsRegister): void {
    // FlashAttention 通过 registry 注册
    // CompositeHandler 通过 operators.find('flash_attention') 获取
    registry.register('flash_attention', flashAttention);
}
