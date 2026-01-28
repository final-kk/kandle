/**
 * attention - 注意力模块
 *
 * 包含两个版本：
 * - MultiheadAttention: 严格对标 PyTorch nn.MultiheadAttention
 * - MultiheadAttentionFast: 便捷推理版本（简化接口）
 */

export {
    MultiheadAttention,
    type MultiheadAttentionOptions,
    type MultiheadAttentionOutput,
} from './MultiheadAttention';

export {
    MultiheadAttentionFast,
    type MultiheadAttentionFastOptions,
    type MultiheadAttentionFastForwardOptions,
} from './MultiheadAttentionFast';
