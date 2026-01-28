/**
 * nn - 神经网络模块
 *
 * 遵循 PyTorch torch.nn 的设计:
 * - nn.functional: 函数式接口 (无状态)
 * - nn.Module: 模块化接口 (有状态)
 * - nn.Parameter: 可学习参数
 * - 容器: Sequential, ModuleList, ModuleDict
 * - 层: Linear, Embedding, LayerNorm, RMSNorm, ReLU, GELU, SiLU, etc.
 *
 * @see https://pytorch.org/docs/stable/nn.html
 */

// ============================================================================
// Functional API
// ============================================================================

// 直接从 ops.ts 导入 functional 对象
import { functional } from '../generated/ops';
export { functional };

// ============================================================================
// Core Classes
// ============================================================================

// Parameter - 可学习参数
export { Parameter, isParameter } from './parameter';

// Module - 基类
export {
    Module,
    type ForwardPreHook,
    type ForwardHook,
    type RemovableHandle,
    type LoadStateDictOptions,
    type LoadStateDictResult,
    type LoadFromSafetensorOptions,
} from './module';

// ============================================================================
// Containers
// ============================================================================

export {
    Sequential,
    ModuleList,
    ModuleDict,
} from './containers';

// ============================================================================
// Modules (层实现)
// ============================================================================

// 线性层
export { Linear, type LinearOptions } from './modules/linear';

// 嵌入层
export { Embedding, type EmbeddingOptions } from './modules/embedding';

// 归一化层
export {
    LayerNorm, type LayerNormOptions,
    RMSNorm, type RMSNormOptions,
} from './modules/normalization';

// 激活函数
export {
    ReLU,
    GELU,
    SiLU,
    Softmax,
    Sigmoid,
    Tanh,
} from './modules/non_linear_activation';

// 注意力模块
export {
    MultiheadAttention,
    type MultiheadAttentionOptions,
    type MultiheadAttentionOutput,
    MultiheadAttentionFast,
    type MultiheadAttentionFastOptions,
    type MultiheadAttentionFastForwardOptions,
} from './modules/attention';

// 卷积层
export { Conv1d, Conv2d, type ConvOptions } from './modules/conv';

// 池化层
export { MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d } from './modules/pooling';
