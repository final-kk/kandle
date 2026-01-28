/**
 * modules - 神经网络模块层
 *
 * 所有 nn.XXX 形式的模块类
 */

// 线性层
export { Linear, type LinearOptions } from './linear';

// 嵌入层
export { Embedding, type EmbeddingOptions } from './embedding';

// 归一化层
export { LayerNorm, type LayerNormOptions, RMSNorm, type RMSNormOptions } from './normalization';

// 激活函数
export { ReLU, GELU, SiLU, Softmax, Sigmoid, Tanh } from './non_linear_activation';

// 注意力模块
export {
    MultiheadAttention,
    type MultiheadAttentionOptions,
    type MultiheadAttentionOutput,
    MultiheadAttentionFast,
    type MultiheadAttentionFastOptions,
    type MultiheadAttentionFastForwardOptions,
} from './attention';

// 卷积层
export { Conv1d, Conv2d, type ConvOptions } from './conv';

// 池化层
export { MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d } from './pooling';

