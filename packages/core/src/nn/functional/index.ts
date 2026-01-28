/**
 * nn.functional - 神经网络函数式接口
 *
 * 从 codegen 生成的 functional 对象重新导出
 * @see https://pytorch.org/docs/stable/nn.functional.html
 */

// 导出 functional 对象本身（包含所有 nn.functional 操作）
export { functional } from '../../generated/ops';

// 同时导出各函数，方便解构导入
// import { relu, sigmoid } from 'kandle/nn/functional'
export {
    // 激活函数
    relu,
    sigmoid,
    tanh,
    gelu,
    silu,
    leakyRelu,
    elu,
    hardtanh,
    selu,
    logsigmoid,

    // Softmax 系列
    softmax,
    logSoftmax,
    softmin,

    // 正则化
    dropout,

    // 归一化
    batchNorm,
    groupNorm,
    layerNorm,
    rmsNorm,
    normalize,

    // 线性
    linear,

    // 嵌入
    embedding,

    // 卷积
    conv1d,
    conv2d,
    conv3d,
    convTranspose2d,

    // 池化
    maxPool1d,
    maxPool2d,
    maxPool3d,
    avgPool1d,
    avgPool2d,
    avgPool3d,
    adaptiveAvgPool2d,
    adaptiveMaxPool2d,
} from '../../generated/ops';

