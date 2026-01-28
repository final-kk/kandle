import { DType, ShapeLike } from "../base";
import { ITensorHandle } from "../tensor";

export interface INNFunctional {

    // ========================================================================
    // Convolution Functions
    // ========================================================================

    /**
    * @description Applies a 1D convolution over an input signal composed of several input planes.
    *
    * @param input Input tensor of shape `(minibatch, in_channels, iW)`.
    * @param weight Filter tensor of shape `(out_channels, in_channels / groups, kW)`.
    * @param bias Optional bias tensor of shape `(out_channels)`. Default: `null`.
    * @param stride The stride of the convolving kernel. Can be a single number or a one-element tuple `(sW,)`. Default: `1`.
    * @param padding Implicit paddings on both sides of the input.
    * Can be a string `{'valid', 'same'}`, single number or a one-element tuple `(padW,)`.
    * Default: `0` (same as no padding).
    * Note: `padding='same'` pads the input so the output has the same shape as the input.
    * However, this mode doesn't support any stride values other than 1.
    * @param dilation The spacing between kernel elements. Can be a single number or a one-element tuple `(dW,)`. Default: `1`.
    * @param groups Split input into groups, `in_channels` should be divisible by the number of groups. Default: `1`.
    * @returns The output tensor.
    */
    conv1d(
        input: ITensorHandle,
        weight: ITensorHandle,
        bias?: ITensorHandle,
        stride?: number | number[],
        padding?: number | number[] | "valid" | "same",
        dilation?: number | number[],
        groups?: number
    ): ITensorHandle;

    /**
    * @description Applies a 2D convolution over an input image composed of several input planes.
    * * This operator **NOT** supports TensorFloat32.
    * * Note: In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance.
    * @param input Input tensor of shape (minibatch, in_channels, iH, iW).
    * @param weight Filters of shape (out_channels, in_channels/groups, kH, kW).
    * @param bias Optional bias tensor of shape (out_channels). Default: null.
    * @param stride The stride of the convolving kernel. Can be a single number or a tuple (sH, sW). Default: 1.
    * @param padding Implicit paddings on both sides of the input. Can be a string {'valid', 'same'}, single number or a tuple (padH, padW). Default: 0. padding='valid' is the same as no padding. padding='same' pads the input so the output has the same shape as the input. However, this mode doesn’t support any stride values other than 1.
    * @param dilation The spacing between kernel elements. Can be a single number or a tuple (dH, dW). Default: 1.
    * @param groups Split input into groups, both in_channels and out_channels should be divisible by the number of groups. Default: 1.
    * @returns The output tensor.
    */
    conv2d(
        input: ITensorHandle,
        weight: ITensorHandle,
        bias?: ITensorHandle,
        stride?: number | number[],
        padding?: number | number[] | "valid" | "same",
        dilation?: number | number[],
        groups?: number
    ): ITensorHandle;

    /**
    * @description Extracts sliding local blocks from a batched input tensor.
    * * Consider a batched `input` tensor of shape (N, C, *), where N is the batch dimension,
    * C is the channel dimension, and * represent arbitrary spatial dimensions.
    * This operation flattens each sliding `kernel_size`-sized block within the spatial dimensions
    * of `input` into a column (i.e., last dimension) of a 3-D `output` tensor.
    * * Warning: Currently, only 4-D input tensors (batched image-like tensors) are supported.
    * * @param input Input tensor of shape (N, C, *).
    * @param kernel_size The size of the sliding blocks.
    * @param dilation A parameter that controls the stride of elements within the neighborhood. Default: 1.
    * @param padding Implicit zero padding to be added on both sides of input. Default: 0.
    * @param stride The stride of the sliding blocks in the input spatial dimensions. Default: 1.
    * @returns The output tensor of shape (N, C * product(kernel_size), L).
    * L is the total number of sliding blocks, typically Output_H * Output_W.
    */
    unfold(
        input: ITensorHandle,
        kernel_size: number | number[],
        dilation?: number | number[],
        padding?: number | number[],
        stride?: number | number[]
    ): ITensorHandle;

    // ========================================================================
    // Pooling functions
    // ========================================================================

    /**
     * @description Applies 2D average-pooling operation in kH * kW regions by step size sH * sW steps.
     * The number of output features is equal to the number of input planes.
     * See AvgPool2d for details and output shape.
     *
     * @param input Input tensor (minibatch, in_channels, iH, iW).
     * @param kernel_size Size of the pooling region. Can be a single number, a single-element tuple or a tuple (kH, kW).
     * @param stride Stride of the pooling operation. Can be a single number, a single-element tuple or a tuple (sH, sW). Default: kernel_size.
     * @param padding Implicit zero paddings on both sides of the input. Can be a single number, a single-element tuple or a tuple (padH, padW). Default: 0.
     * @param ceil_mode When True, will use ceil instead of floor in the formula to compute the output shape. Default: false.
     * @param count_include_pad When True, will include the zero-padding in the averaging calculation. Default: true.
     * @param divisor_override If specified, it will be used as divisor, otherwise size of the pooling region will be used. Default: null.
     * @returns The output tensor.
     */
    avgPool2d(
        input: ITensorHandle,
        kernel_size: number | number[],
        stride?: number | number[],
        padding?: number | number[],
        ceil_mode?: boolean,
        count_include_pad?: boolean,
        divisor_override?: number
    ): ITensorHandle;

    /**
     * @description Applies a 2D max pooling over an input signal composed of several input planes.
     * See MaxPool2d for details.
     *
     * @param input Input tensor (minibatch, in_channels, iH, iW), minibatch dim optional.
     * @param kernel_size Size of the pooling region. Can be a single number or a tuple (kH, kW).
     * @param stride Stride of the pooling operation. Can be a single number or a tuple (sH, sW). Default: kernel_size.
     * @param padding Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2. Default: 0.
     * @param dilation The stride between elements within a sliding window, must be > 0. Default: 1.
     * @param ceil_mode If True, will use ceil instead of floor to compute the output shape. This ensures that every element in the input tensor is covered by a sliding window. Default: false.
     * @param return_indices If True, will return the argmax along with the max values. Useful for torch.nn.functional.max_unpool2d later. Default: false.
     * @returns The output tensor, or a tuple (output, indices) if return_indices is true.
     */
    maxPool2d(
        input: ITensorHandle,
        kernel_size: number | number[],
        stride?: number | number[],
        padding?: number | number[],
        dilation?: number | number[],
        ceil_mode?: boolean,
        return_indices?: false
    ): ITensorHandle

    maxPool2d(
        input: ITensorHandle,
        kernel_size: number | number[],
        stride?: number | number[],
        padding?: number | number[],
        dilation?: number | number[],
        ceil_mode?: boolean,
        return_indices?: true
    ): [ITensorHandle, ITensorHandle];

    // ========================================================================
    // Non-linear activation functions
    // ========================================================================

    /**
     * @description Applies the rectified linear unit function element-wise.
     * ReLU(x) = max(0, x)
     * @param input Input tensor of arbitrary shape.
     * @param inplace Can optionally do the operation in-place. Default: false.
     * @returns The output tensor of the same shape as input.
     */
    relu(
        input: ITensorHandle,
        inplace?: boolean
    ): ITensorHandle;

    // relu6
    // selu

    /**
     * @description Applies the Gaussian Error Linear Unit (GELU) function element-wise.
     * * When `approximate` is 'none', it applies: GELU(x) = x * Φ(x), where Φ(x) is the Cumulative Distribution Function for Gaussian Distribution.
     * * When `approximate` is 'tanh', it estimates GELU with: GELU(x) = 0.5 * x * (1 + Tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
     * @param input Input tensor.
     * @param approximate The gelu approximation algorithm to use: 'none' | 'tanh'. Default: 'none'.
     * @returns The output tensor.
     */
    gelu(
        input: ITensorHandle,
        approximate?: "none" | "tanh"
    ): ITensorHandle;

    /**
     * @description Applies the Logsigmoid function element-wise.
     *
     * Formula:
     * $$ \text{LogSigmoid}(x) = \log\left(\frac{ 1 }{ 1 + \exp(-x)}\right) $$
     *
     * Shape:
     * - Input: (*), where * means any number of dimensions.
     * - Output: (*), same shape as the input.
     *
     * @param input Input tensor.
     * @returns The output tensor.
     */
    logSigmoid(input: ITensorHandle): ITensorHandle;

    /**
     * @description Apply a softmax function.
     *
     * Softmax is defined as:
     * $$ \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)} $$
     *
     * It is applied to all slices along dim, and will re-scale them so that the elements
     * lie in the range [0, 1] and sum to 1.
     *
     * Note: This function doesn’t work directly with NLLLoss, which expects the Log to be computed between the Softmax and itself.
     * Use log_softmax instead (it’s faster and has better numerical properties).
     *
     * @param input Input tensor.
     * @param dim A dimension along which softmax will be computed. default: -1 (the last dimension).
     * @param dtype The desired data type of returned tensor. If specified, the input tensor is casted to dtype before the operation is performed. Default: null.
     * @returns The output tensor.
     */
    softmax(
        input: ITensorHandle,
        dim?: number,
        dtype?: DType
    ): ITensorHandle;

    /**
     * @description Applies a softmax followed by a logarithm.
     *
     * While mathematically equivalent to $\log(\text{Softmax}(x))$, doing these two operations separately is slower and numerically unstable. This function uses an alternative formulation to compute the output and gradient correctly.
     *
     * Formula:
     * $$ \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right) $$
     *
     * Shape:
     * - Input: (*), where * means any number of additional dimensions.
     * - Output: (*), same shape as the input.
     *
     * @param input Input tensor.
     * @param dim A dimension along which log_softmax will be computed.
     * @param dtype The desired data type of returned tensor. If specified, the input tensor is cast to dtype before the operation is performed. This is useful for preventing data type overflows. Default: null.
     * @returns The output tensor.
     */
    logSoftmax(
        input: ITensorHandle,
        dim?: number,
        dtype?: DType
    ): ITensorHandle;

    /**
     * @description Applies the Sigmoid function element-wise.
     *
     * Formula:
     * $$ \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)} $$
     *
     * Shape:
     * - Input: (*), where * means any number of dimensions.
     * - Output: (*), same shape as the input.
     *
     * @param input Input tensor.
     * @returns The output tensor.
     */
    sigmoid(input: ITensorHandle): ITensorHandle;

    /**
     * @description Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
     * The SiLU function is also known as the swish function.
     *
     * Formula:
     * $$ \text{silu}(x) = x * \sigma(x) $$
     * where $\sigma(x)$ is the logistic sigmoid.
     *
     * Shape:
     * - Input: (*), where * means any number of dimensions.
     * - Output: (*), same shape as the input.
     *
     * @param input Input tensor.
     * @param inplace can optionally do the operation in-place. Default: false.
     * @returns The output tensor.
     */
    silu(input: ITensorHandle, inplace?: boolean): ITensorHandle;

    /**
     * @description Apply Batch Normalization for each channel across a batch of data.
     * See BatchNorm1d, BatchNorm2d, BatchNorm3d for details.
     *
     * @param input The input tensor.
     * @param running_mean The running mean tensor.
     * @param running_var The running variance tensor.
     * @param weight The weight tensor (gamma). Default: null.
     * @param bias The bias tensor (beta). Default: null.
     * @param training A boolean value that when set to True, this module uses batch statistics. Default: false.
     * @param momentum The value used for the running_mean and running_var computation. Default: 0.1.
     * @param eps A value added to the denominator for numerical stability. Default: 1e-05.
     * @returns The output tensor.
     */
    batchNorm(
        input: ITensorHandle,
        running_mean?: ITensorHandle,
        running_var?: ITensorHandle,
        weight?: ITensorHandle,
        bias?: ITensorHandle,
        training?: boolean,
        momentum?: number,
        eps?: number
    ): ITensorHandle;

    /**
     * @description Applies Group Normalization over a mini-batch of inputs.
     * The input channels are separated into num_groups groups, each containing num_channels / num_groups channels.
     * The mean and standard-deviation are calculated separately over each group.
     * @param input The input tensor.
     * @param numGroups Number of groups to separate the channels into.
     * @param weight Optional learnable per-channel affine weight (gamma). Default: null.
     * @param bias Optional learnable per-channel affine bias (beta). Default: null.
     * @param eps A value added to the denominator for numerical stability. Default: 1e-05.
     * @returns The output tensor.
     */
    groupNorm(
        input: ITensorHandle,
        numGroups: number,
        weight?: ITensorHandle,
        bias?: ITensorHandle,
        eps?: number
    ): ITensorHandle;

    /**
     * @description Applies Layer Normalization over a mini-batch of inputs.
     * The mean and standard-deviation are calculated over the last D dimensions, where D is the dimension of normalized_shape.
     * @param input The input tensor.
     * @param normalizedShape Input shape from an expected input of size. If a single integer is used, it is treated as a singleton list.
     * @param weight Optional learnable per-element affine weight (gamma). Default: null.
     * @param bias Optional learnable per-element affine bias (beta). Default: null.
     * @param eps A value added to the denominator for numerical stability. Default: 1e-05.
     * @returns The output tensor.
     */
    layerNorm(
        input: ITensorHandle,
        normalizedShape: number | number[],
        weight?: ITensorHandle,
        bias?: ITensorHandle,
        eps?: number
    ): ITensorHandle;

    /**
     * @description Applies Root Mean Square Layer Normalization.
     * The RMS is taken over the last D dimensions, where D is the dimension of normalized_shape.
     * @param input The input tensor.
     * @param normalizedShape Input shape from an expected input of size. If a single integer is used, it is treated as a singleton list.
     * @param weight Optional learnable per-element affine weight (gamma). Default: null.
     * @param eps A value added to the denominator for numerical stability. Default: torch.finfo(x.dtype).eps.
     * @returns The output tensor.
     */
    rmsNorm(
        input: ITensorHandle,
        normalizedShape: number | number[],
        weight?: ITensorHandle,
        eps?: number
    ): ITensorHandle;

    /**
     * @description Perform L_p normalization of inputs over specified dimension.
     * For a tensor input of sizes (n0, ..., ndim, ..., nk), each ndim-element vector v along dimension dim is transformed as:
     * v = v / max(||v||_p, eps).
     * With the default arguments it uses the Euclidean norm over vectors along dimension 1 for normalization.
     * @param input Input tensor of any shape.
     * @param p The exponent value in the norm formulation. Default: 2.0.
     * @param dim The dimension to reduce. Default: 1.
     * @param eps Small value to avoid division by zero. Default: 1e-12.
     * @param out The output tensor. If out is used, this operation won’t be differentiable. Default: null.
     * @returns The output tensor.
     */
    normalize(
        input: ITensorHandle,
        p?: number,
        dim?: number | number[],
        eps?: number,
        out?: ITensorHandle
    ): ITensorHandle;

    // ========================================================================
    // Linear functions
    // ========================================================================

    /**
     * @description Applies a linear transformation to the incoming data: y = xA^T + b.
     *
     * This operation supports 2-D weight with sparse layout.
     *
     * Shape:
     * - Input: (*, in_features)
     * - Weight: (out_features, in_features) or (in_features)
     * - Bias: (out_features) or ()
     * - Output: (*, out_features) or (*)
     *
     * @param input Input tensor of shape (*, in_features).
     * @param weight Weights of shape (out_features, in_features) or (in_features).
     * @param bias Optional bias tensor of shape (out_features) or (). Default: null.
     * @returns The output tensor.
     */
    linear(
        input: ITensorHandle,
        weight: ITensorHandle,
        bias?: ITensorHandle
    ): ITensorHandle;


    // ========================================================================
    // Sparse functions
    // ========================================================================

    /**
     * @description Generate a simple lookup table that looks up embeddings in a fixed dictionary and size.
     *
     * This module is often used to retrieve word embeddings using indices. The input to the module is a list of indices, and the embedding matrix, and the output is the corresponding word embeddings.
     *
     * Shape:
     * - Input: LongTensor of arbitrary shape containing the indices to extract (*).
     * - Weight: Embedding matrix of floating point type with shape (V, embedding_dim), where V = maximum index + 1 and embedding_dim = the embedding size.
     * - Output: (*, embedding_dim), where * is the input shape.
     *
     * Note: The analytical gradients of this function with respect to entries in weight at the row specified by padding_idx are expected to differ from the numerical ones.
     * Note: This will modify weight in-place if max_norm is given.
     *
     * @param input Tensor containing indices into the embedding matrix.
     * @param weight The embedding matrix with number of rows equal to the maximum possible index + 1, and number of columns equal to the embedding size.
     * @param padding_idx If specified, the entries at padding_idx do not contribute to the gradient; therefore, the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”. Default: null.
     * @param max_norm If given, each embedding vector with norm larger than max_norm is renormalized to have norm max_norm. Note: this will modify weight in-place. Default: null.
     * @param norm_type The p of the p-norm to compute for the max_norm option. Default: 2.0.
     * @param scale_grad_by_freq If given, this will scale gradients by the inverse of frequency of the words in the mini-batch. Default: false.
     * @param sparse If True, gradient w.r.t. weight will be a sparse tensor. Default: false.
     * @returns The output tensor.
     */
    embedding(
        input: ITensorHandle,
        weight: ITensorHandle,
        padding_idx?: number,
        max_norm?: number,
        norm_type?: number,
        scale_grad_by_freq?: boolean,
        sparse?: boolean
    ): ITensorHandle;

}
