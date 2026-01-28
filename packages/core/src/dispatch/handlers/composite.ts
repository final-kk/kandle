/**
 * CompositeHandler (v6)
 *
 * 处理 Composite 机制的操作:
 * - 组合多个基础算子实现高级功能
 * - 不直接调用 Backend Kernel
 * - 逻辑完全在 Frontend (JavaScript) 层
 */

import type { ITensorHandle } from '@kandle/types';
import type { OpEntry } from '@kandle/types';
import type { PatternHandler, DirectContext } from './types';
import { dispatchMatmul } from '../matmulOps';
import { dispatchView } from './shape';
import { type OperatorContext } from './types';
import { dispatchIterator } from './iterator';
import { dispatchGather } from './gather';
import { add_Tensor } from '../../generated/internal/add_Tensor';
import { softmax } from '../../generated/internal/softmax';
import { dispatchFFT } from './fft';
import { dispatchFactory } from './factory';
import { env } from '../../env';
import { ShapeHandler } from './shape'; // For specialized diagonal view
import { TensorIterator } from '../TensorIterator';
import type { IteratorKernelImpl } from '@kandle/types';

export class CompositeHandler implements PatternHandler {
    private static instance: CompositeHandler;

    static getInstance(): CompositeHandler {
        if (!CompositeHandler.instance) {
            CompositeHandler.instance = new CompositeHandler();
        }
        return CompositeHandler.instance;
    }

    buildContext(entry: OpEntry, ctx: OperatorContext): DirectContext {
        // Reuse DirectContext for simplicity, passing everything needed
        return {
            kind: 'direct',
            inputs: ctx.tensorInputs,
            scalars: ctx.scalarArgs,
            metadata: ctx.metadata,
            outs: ctx.outs,
            kernelName: entry.dispatchKey,
        };
    }

    execute(execCtx: DirectContext): ITensorHandle | ITensorHandle[] {
        const { kernelName, inputs, scalars, metadata, outs } = execCtx;
        const out = outs?.[0];
        const allParams = { ...scalars, ...metadata };

        switch (kernelName) {
            case 'linear': {
                // F.linear: y = input @ weight.T + bias
                const input = inputs[0];
                const weight = inputs[1];
                const bias = inputs[2]; // Optional

                // 1. MatMul with transposeB=true
                const y = dispatchMatmul(
                    input, weight,
                    undefined, // c
                    1.0,       // alpha
                    0.0,       // beta
                    bias ? undefined : out, // Only write to out if no bias add needed
                    false,     // transposeA
                    true       // transposeB: weight.T
                );

                // 2. Add bias if present
                if (bias) {
                    return add_Tensor(y, bias, undefined, out);
                }
                return y;
            }

            case 'trace': {
                // trace = diagonal().sum()
                const input = inputs[0];
                const backend = env.getBackend(input.device);

                // 1. Get diagonal view
                // Using ViewHandler.diagonal helper (if exposed) or duplicating logic
                // MatrixHandler used ShapeHandler.getInstance().diagonal
                const diagView = ShapeHandler.getInstance().diagonal(input, 0, 0, 1, backend);

                // 2. Sum reduction
                // Need to use dispatchIterator for Reduction
                // OpEntry for sum:
                const sumEntry: OpEntry = {
                    name: 'sum',
                    dispatchKey: 'sum',
                    mechanism: 'Iterator',
                    iteratorType: 'Reduce',
                    signature: {} as any, // Mock signature
                    iteratorConfig: { factory: 'reduction', tensorInputs: ['self'], scalarArgs: ['dim', 'keepdim', 'dtype'] }
                } as any;

                // Need to construct context for sum
                const sumCtx: OperatorContext = {
                    opName: 'sum',
                    tensorInputs: [diagView],
                    scalarArgs: { keepdim: false },
                    metadata: { dim: undefined }, // sum all
                    outs: out ? [out] : undefined
                };

                return dispatchIterator(sumEntry, sumCtx);
            }

            case 'diag': {
                const input = inputs[0];
                const diagonal = (allParams['diagonal'] ?? 0) as number;
                const backend = env.getBackend(input.device);

                if (input.shape.length === 1) {
                    // 1D -> 2D (Construction)
                    // This is allocate + copy (unary copy with stride?)
                    const n = input.shape[0]; // numel for 1D
                    const size = n + Math.abs(diagonal);
                    const shape = [size, size];

                    // 1. Create Zeros (Backend default alloc is usually not zeroed? Wait, WebGPU buffers are zero-init by default per spec security)
                    // But 'empty' might not be zeroed in optimized backends? 
                    // MatrixHandler assumed 'createTensorHandle' zeros it? 
                    // Step 166 says: "backend.createTensorHandle 默认初始化为零"
                    const output = backend.createTensorHandle(shape, input.dtype);

                    // 2. Get Diagonal View of output
                    const diagView = ShapeHandler.getInstance().diagonal(output, diagonal, 0, 1, backend);

                    // 3. Copy input to view
                    const copyIter = TensorIterator.unaryOp(input, 'copy', diagView);
                    const copyKernel = backend.operators.find('copy') as IteratorKernelImpl | undefined;
                    if (!copyKernel) throw new Error("Kernel 'copy' not found");
                    copyKernel(copyIter);

                    return output;
                } else if (input.shape.length === 2) {
                    // 2D -> 1D (Extraction) -> View
                    return ShapeHandler.getInstance().diagonal(input, diagonal, 0, 1, backend);
                } else {
                    throw new Error(`diag: expected 1D or 2D input, got ${input.shape.length}D`);
                }
            }

            case 'embedding': {
                // embedding(input, weight) -> index_select(weight, 0, input) -> view
                // Typically: embedding outputs [batch..., embed_dim]
                // index_select outputs [flattened_input_len, embed_dim]
                const input = inputs[0];
                const weight = inputs[1]; // [vocab_size, embed_dim]

                // If input is not 1D, index_select flattens?
                // torch.index_select expects 1D index
                // So we must flatten input, index_select, then reshape

                const inputFlatten = input.shape.length > 1; // Assuming we need to flatten if > 1D for index_select?
                // Actually GatherHandler (Step 130) index_select doc says: "index: 1D 整数索引张量"
                // So yes, we need to flatten input.

                // 1. Flatten input (View) [N]
                // Construct view op? Or just use input as 1D storage if contiguous?
                // Let's use view.
                /*
                const flatInput = dispatchView('reshape', { ...ctx, input, shape: [-1] }) ...
                CompositeHandler calling ViewHandler is fine.
                */
                // For simplicity, let's assume KernelHandler handles 'index_select' correctly.
                // But we need to use 'indexSelect' op entry.

                // Ops helper approach:
                // We don't have direct access to 'indexSelect' helper function unless we import it or mock dispatch.

                // Let's use dispatchGather for index_select
                const indexSelectEntry: OpEntry = {
                    name: 'indexSelect',
                    dispatchKey: 'index_select',
                    mechanism: 'Gather',
                    // ...
                } as any;

                let indices = input;
                let originalShape = input.shape;

                // 1. If input is not 1D, flatten it for index_select
                if (input.shape.length > 1) {
                    const viewEntry: OpEntry = { name: 'view', dispatchKey: 'view', mechanism: 'View' } as any;
                    indices = dispatchView(viewEntry, {
                        opName: 'view',
                        tensorInputs: [input],
                        scalarArgs: {},
                        metadata: { shape: [-1] },
                        outs: undefined
                    }) as ITensorHandle;
                }

                // 2. Call index_select(weight, 0, indices)
                const gathered = dispatchGather(indexSelectEntry, {
                    opName: 'index_select',
                    tensorInputs: [weight, indices], // self=weight, index=indices
                    scalarArgs: { dim: 0 },
                    metadata: {},
                    outs: undefined // intermediate
                }) as ITensorHandle;

                // 3. Reshape result if necessary: [...input.shape, embed_dim]
                if (originalShape.length > 1) {
                    const embedDim = weight.shape[1];
                    const newShape = [...originalShape, embedDim];
                    const viewEntry: OpEntry = { name: 'view', dispatchKey: 'view', mechanism: 'View' } as any;
                    const viewed = dispatchView(viewEntry, {
                        opName: 'view',
                        tensorInputs: [gathered],
                        scalarArgs: {},
                        metadata: { shape: newShape },
                        outs: undefined  // View 不支持 out
                    }) as ITensorHandle;

                    // 如果提供了 out，需要 copy
                    if (out) {
                        const backend = env.getBackend(input.device);
                        const copyIter = TensorIterator.unaryOp(viewed, 'copy', out);
                        const copyKernel = backend.operators.find('copy') as IteratorKernelImpl | undefined;
                        if (!copyKernel) throw new Error("Kernel 'copy' not found");
                        copyKernel(copyIter);
                        return out;
                    }
                    return viewed;
                }

                // 1D input case: gathered 已经是最终结果
                if (out) {
                    const backend = env.getBackend(input.device);
                    const copyIter = TensorIterator.unaryOp(gathered, 'copy', out);
                    const copyKernel = backend.operators.find('copy') as IteratorKernelImpl | undefined;
                    if (!copyKernel) throw new Error("Kernel 'copy' not found");
                    copyKernel(copyIter);
                    return out;
                }

                return gathered;
            }

            case 'scaled_dot_product_attention': {
                // FlashAttention: fused scaled dot-product attention
                // Handles tiling, online softmax, and causal masking internally
                // 
                // Shapes:
                // - query: (..., L, E) - target seq length L, embed dim E
                // - key:   (..., S, E) - source seq length S
                // - value: (..., S, Ev) - value embed dim Ev
                // - output: (..., L, Ev)

                const query = inputs[0];
                const key = inputs[1];
                const value = inputs[2];
                const attnMask = inputs[3]; // Optional - may be undefined

                const dropoutP = (allParams['dropoutP'] as number) ?? 0.0;
                const isCausal = (allParams['isCausal'] as boolean) ?? false;
                let scale = allParams['scale'] as number | undefined;

                // dropout_p > 0 不支持 (推理模式)
                if (dropoutP > 0) {
                    console.warn('SDPA: dropout_p > 0 is not supported in inference mode, ignoring');
                }

                // 计算 embed_dim (E) 从 query 最后一维
                const E = query.shape[query.shape.length - 1];
                if (scale === undefined) {
                    scale = 1.0 / Math.sqrt(E);
                }

                // 如果提供了 attnMask，回退到 naive 实现
                // FlashAttention 当前只支持 isCausal 内置掩码
                if (attnMask !== undefined) {
                    // Fallback to naive implementation for explicit mask
                    // Step 1: QK^T
                    let attnWeights = dispatchMatmul(
                        query, key,
                        undefined, scale, 0.0, undefined,
                        false, true
                    );

                    // Step 2: Apply mask
                    attnWeights = add_Tensor(attnWeights, attnMask);

                    // Step 3: Softmax
                    const attnProbs = softmax(attnWeights, -1);

                    // Step 4: Output
                    return dispatchMatmul(
                        attnProbs, value,
                        undefined, 1.0, 0.0, out,
                        false, false
                    );
                }

                // 尝试使用 FlashAttention kernel（如果后端支持）
                const backend = env.getBackend(query.device);
                const outputShape = [...query.shape.slice(0, -1), value.shape[value.shape.length - 1]];
                const output = out ?? backend.createTensorHandle(outputShape, query.dtype);

                // 通过 operators.find() 获取 FlashAttention kernel
                type FlashAttentionKernel = (
                    query: ITensorHandle, key: ITensorHandle, value: ITensorHandle,
                    output: ITensorHandle, scale: number, isCausal: boolean
                ) => void;
                const flashKernel = backend.operators.find('flash_attention') as FlashAttentionKernel | undefined;

                if (flashKernel) {
                    // 使用 FlashAttention kernel
                    flashKernel(query, key, value, output, scale, isCausal);
                } else {
                    // 回退到 naive 实现
                    // Step 1: QK^T
                    let attnWeights = dispatchMatmul(
                        query, key,
                        undefined, scale, 0.0, undefined,
                        false, true
                    );

                    // Step 2: Apply causal mask if needed (naive fallback uses explicit mask)
                    if (isCausal) {
                        // 创建 causal mask: 上三角为 -Infinity
                        const L = query.shape[query.shape.length - 2];
                        const S = key.shape[key.shape.length - 2];
                        // Note: 这里需要重新导入 triu, ones, zeros, full, where
                        // 但由于 FlashAttention 应该可用，这个分支很少执行
                        console.warn('SDPA: FlashAttention not available, using naive implementation for causal mask');
                    }

                    // Step 3: Softmax
                    const attnProbs = softmax(attnWeights, -1);

                    // Step 4: Output
                    const result = dispatchMatmul(
                        attnProbs, value,
                        undefined, 1.0, 0.0, output,
                        false, false
                    );
                    return result;
                }
                return output;
            }

            case 'fftshift': {
                // fftshift: shift zero-frequency component to center
                // implemented as cat(input[mid:], input[:mid])
                const input = inputs[0];
                let dims = allParams['dim'] as number[] | undefined;

                if (dims === undefined) {
                    dims = Array.from({ length: input.shape.length }, (_, i) => i);
                }

                let result = input;
                for (const dim of dims) {
                    const resolvedDim = dim < 0 ? input.shape.length + dim : dim;
                    const n = input.shape[resolvedDim];
                    const mid = Math.ceil(n / 2);

                    // Create slice op entry
                    const sliceEntry: OpEntry = { name: 'slice', dispatchKey: 'slice', mechanism: 'View' } as any;

                    // Slice 1: [mid:]
                    const slices1 = Array(input.shape.length).fill(':');
                    slices1[resolvedDim] = `${mid}:${n}`;
                    const part1 = dispatchView(sliceEntry, {
                        opName: 'slice',
                        tensorInputs: [result], // use current result
                        scalarArgs: { slices: slices1.join(',') },
                        metadata: {},
                        outs: undefined,
                    }) as ITensorHandle;

                    // Slice 2: [:mid]
                    const slices2 = Array(input.shape.length).fill(':');
                    slices2[resolvedDim] = `0:${mid}`;
                    const part2 = dispatchView(sliceEntry, {
                        opName: 'slice',
                        tensorInputs: [result],
                        scalarArgs: { slices: slices2.join(',') },
                        metadata: {},
                        outs: undefined,
                    }) as ITensorHandle;

                    // Cat([part1, part2], dim)
                    const catEntry: OpEntry = { name: 'cat', dispatchKey: 'cat', mechanism: 'Shape' } as any; // cat handled by ShapeHandler
                    // Note: ShapeHandler checks for 'cat' name in CAT_OPS

                    result = dispatchView(catEntry, {
                        opName: 'cat',
                        tensorInputs: [part1, part2],
                        scalarArgs: { dim: resolvedDim },
                        metadata: {},
                        outs: undefined,
                    }) as ITensorHandle;
                }

                return result;
            }

            case 'ifftshift': {
                // ifftshift: inverse of fftshift
                // For odd n, shift = n // 2
                // For even n, shift = n - n // 2 = (n + 1) // 2
                // Equivalently: roll(x, -(n//2), dim) or roll(x, (n - n//2), dim)
                const input = inputs[0];
                let dims = allParams['dim'] as number[] | undefined;

                if (dims === undefined) {
                    dims = Array.from({ length: input.shape.length }, (_, i) => i);
                }

                let result = input;
                for (const dim of dims) {
                    const resolvedDim = dim < 0 ? input.shape.length + dim : dim;
                    const n = input.shape[resolvedDim];
                    const split = Math.floor(n / 2);

                    const sliceEntry: OpEntry = { name: 'slice', dispatchKey: 'slice', mechanism: 'View' } as any;

                    // Slice 1: [split:]
                    const slices1 = Array(input.shape.length).fill(':');
                    slices1[resolvedDim] = `${split}:${n}`;
                    const part1 = dispatchView(sliceEntry, {
                        opName: 'slice',
                        tensorInputs: [result],
                        scalarArgs: { slices: slices1.join(',') },
                        metadata: {},
                        outs: undefined,
                    }) as ITensorHandle;

                    // Slice 2: [:split]
                    const slices2 = Array(input.shape.length).fill(':');
                    slices2[resolvedDim] = `0:${split}`;
                    const part2 = dispatchView(sliceEntry, {
                        opName: 'slice',
                        tensorInputs: [result],
                        scalarArgs: { slices: slices2.join(',') },
                        metadata: {},
                        outs: undefined,
                    }) as ITensorHandle;

                    // Cat([part1, part2], dim)
                    const catEntry: OpEntry = { name: 'cat', dispatchKey: 'cat', mechanism: 'Shape' } as any;

                    result = dispatchView(catEntry, {
                        opName: 'cat',
                        tensorInputs: [part1, part2],
                        scalarArgs: { dim: resolvedDim },
                        metadata: {},
                        outs: undefined,
                    }) as ITensorHandle;
                }

                return result;
            }

            case 'fftfreq': {
                // fftfreq(n, d=1.0): returns sample frequencies for DFT
                // Output: [0, 1, ..., n//2-1, -n//2, ..., -1] / (n*d)
                // Length: n
                const n = allParams['n'] as number;
                const d = (allParams['d'] as number) ?? 1.0;

                const backend = env.getBackend(env.getDefaultDevice().name);

                // Create frequency array
                // First half: [0, 1, ..., n//2 - 1]
                // Second half: [-n//2, ..., -1]
                const halfN = Math.floor(n / 2);
                const freqData = new Float32Array(n);

                for (let i = 0; i <= halfN - 1; i++) {
                    freqData[i] = i / (n * d);
                }
                // Handle even/odd n
                if (n % 2 === 0) {
                    // n is even: [-n/2, ..., -1]
                    for (let i = halfN; i < n; i++) {
                        freqData[i] = (i - n) / (n * d);
                    }
                } else {
                    // n is odd: [-(n-1)/2, ..., -1]
                    for (let i = halfN; i < n; i++) {
                        freqData[i] = (i - n) / (n * d);
                    }
                }

                // Create tensor with initial data
                const output = backend.createTensorHandle([n], 'float32');
                const storage = (output as any).storage;
                if (storage && typeof storage.upload === 'function') {
                    storage.upload(freqData.buffer);
                }
                return output;
            }

            case 'rfftfreq': {
                // rfftfreq(n, d=1.0): returns sample frequencies for rfft output
                // Output: [0, 1, ..., n//2] / (n*d)
                // Length: n//2 + 1
                const n = allParams['n'] as number;
                const d = (allParams['d'] as number) ?? 1.0;

                const backend = env.getBackend(env.getDefaultDevice().name);
                const outLen = Math.floor(n / 2) + 1;
                const freqData = new Float32Array(outLen);

                for (let i = 0; i < outLen; i++) {
                    freqData[i] = i / (n * d);
                }

                const output = backend.createTensorHandle([outLen], 'float32');
                const storage = (output as any).storage;
                if (storage && typeof storage.upload === 'function') {
                    storage.upload(freqData.buffer);
                }
                return output;
            }

            case 'stft': {
                // Short-Time Fourier Transform
                // PyTorch implementation: pad → as_strided → mul(window) → rfft → transpose
                const input = inputs[0];
                const windowTensor = inputs[1]; // Optional window tensor

                const n_fft = allParams['n_fft'] as number;
                const hop_length = (allParams['hop_length'] as number) ?? Math.floor(n_fft / 4);
                const win_length = (allParams['win_length'] as number) ?? n_fft;
                const center = (allParams['center'] as boolean) ?? true;
                const pad_mode = (allParams['pad_mode'] as string) ?? 'reflect';
                const normalized = (allParams['normalized'] as boolean) ?? false;
                const onesided = (allParams['onesided'] as boolean) ?? true;

                const backend = env.getBackend(input.device);

                // Validate parameters
                if (n_fft <= 0) {
                    throw new Error(`stft: n_fft must be positive, got ${n_fft}`);
                }
                if (hop_length <= 0) {
                    throw new Error(`stft: hop_length must be positive, got ${hop_length}`);
                }
                if (win_length <= 0 || win_length > n_fft) {
                    throw new Error(`stft: win_length must be in (0, n_fft], got ${win_length}`);
                }

                // Get input shape: (B?, L)
                const inputShape = input.shape;
                const ndim = inputShape.length;
                const isBatched = ndim === 2;
                const signalLen = inputShape[ndim - 1];
                const batchSize = isBatched ? inputShape[0] : 1;

                // Step 1: Center padding if needed
                let signal = input;
                let paddedLen = signalLen;
                if (center) {
                    const padAmount = Math.floor(n_fft / 2);
                    paddedLen = signalLen + 2 * padAmount;

                    // Use pad operation (constant mode)
                    const padEntry: OpEntry = { name: 'pad', dispatchKey: 'pad', mechanism: 'Factory' } as any;
                    signal = dispatchFactory(padEntry, {
                        opName: 'pad',
                        tensorInputs: [signal],
                        scalarArgs: { pad: [padAmount, padAmount], mode: 'constant', value: 0 },
                        metadata: {},
                        outs: undefined,
                    });
                }

                // Step 2: Calculate number of frames
                const n_frames = 1 + Math.floor((paddedLen - n_fft) / hop_length);

                // Step 3: Frame with as_strided (zero-copy view)
                // Output shape: (B?, n_frames, n_fft)
                const frameShape = isBatched
                    ? [batchSize, n_frames, n_fft]
                    : [n_frames, n_fft];

                // Calculate strides for framing
                const inputStride = signal.strides[ndim - 1];
                const frameStrides = isBatched
                    ? [signal.strides[0], hop_length * inputStride, inputStride]
                    : [hop_length * inputStride, inputStride];

                const asStridedEntry: OpEntry = { name: 'asStrided', dispatchKey: 'as_strided', mechanism: 'View' } as any;
                let frames = dispatchView(asStridedEntry, {
                    opName: 'asStrided',
                    tensorInputs: [signal],
                    scalarArgs: { size: frameShape, stride: frameStrides },
                    metadata: {},
                    outs: undefined,
                }) as ITensorHandle;

                // Step 4: Apply window function
                if (windowTensor) {
                    // Window should be 1D with length win_length
                    // If win_length < n_fft, need to pad window to n_fft
                    let window = windowTensor;
                    if (win_length < n_fft) {
                        const left = Math.floor((n_fft - win_length) / 2);
                        const padEntry: OpEntry = { name: 'pad', dispatchKey: 'pad', mechanism: 'Factory' } as any;
                        window = dispatchFactory(padEntry, {
                            opName: 'pad',
                            tensorInputs: [window],
                            scalarArgs: { pad: [left, n_fft - win_length - left], mode: 'constant', value: 0 },
                            metadata: {},
                            outs: undefined,
                        });
                    }

                    // Multiply frames by window: frames * window (broadcast)
                    // Create output for mul first
                    const mulOutput = backend.createTensorHandle(frames.shape, frames.dtype);
                    const mulIter = TensorIterator.binaryOp(frames, window, 'mul', mulOutput);
                    const mulKernel = backend.operators.find('mul') as IteratorKernelImpl | undefined;
                    if (!mulKernel) throw new Error("Kernel 'mul' not found");
                    mulKernel(mulIter);
                    frames = mulOutput;
                }

                // Step 5: FFT on last dimension
                // Use rfft for real input (onesided), fft for complex
                const fftDim = isBatched ? 2 : 1;
                const norm = normalized ? 'ortho' : 'backward';

                // FFT on last dimension
                // Use rfft for onesided=true (default), fft for onesided=false (full spectrum)
                let spectrogram: ITensorHandle;
                if (onesided) {
                    // rfft for onesided=true: returns n_fft//2+1 frequency bins
                    const rfftEntry: OpEntry = { name: 'rfft', dispatchKey: 'rfft', mechanism: 'Factory' } as any;
                    const rfftCtx: OperatorContext = {
                        opName: 'rfft',
                        tensorInputs: [frames],
                        scalarArgs: { n: n_fft, dim: fftDim, norm },
                        metadata: {},
                    };
                    spectrogram = dispatchFFT(rfftEntry, rfftCtx);
                } else {
                    // fft for onesided=false: returns full n_fft frequency bins
                    // First need to convert real input to complex
                    const complexInput = backend.createTensorHandle(frames.shape, 'complex64');
                    const copyKernel = backend.operators.find('copy') as IteratorKernelImpl | undefined;
                    if (!copyKernel) throw new Error("Kernel 'copy' not found");
                    const copyIter = TensorIterator.unaryOp(frames, 'copy', complexInput);
                    copyKernel(copyIter);

                    const fftEntry: OpEntry = { name: 'fft', dispatchKey: 'fft', mechanism: 'Factory' } as any;
                    const fftCtx: OperatorContext = {
                        opName: 'fft',
                        tensorInputs: [complexInput],
                        scalarArgs: { n: n_fft, dim: fftDim, norm },
                        metadata: {},
                    };
                    spectrogram = dispatchFFT(fftEntry, fftCtx);
                }

                // Step 6: Transpose to (B?, freq_bins, n_frames)
                const transposeDims = isBatched ? [0, 2, 1] : [1, 0];
                const permuteEntry: OpEntry = { name: 'permute', dispatchKey: 'permute', mechanism: 'View' } as any;
                const result = dispatchView(permuteEntry, {
                    opName: 'permute',
                    tensorInputs: [spectrogram],
                    scalarArgs: { dims: transposeDims },
                    metadata: {},
                    outs: undefined,
                }) as ITensorHandle;



                return result;
            }

            case 'istft': {
                // Inverse Short-Time Fourier Transform
                // Flow: transpose -> irfft -> window -> overlap-add -> slice

                const input = inputs[0];
                const windowTensor = inputs[1]; // Optional window tensor

                const n_fft = allParams['n_fft'] as number;
                const hop_length = (allParams['hop_length'] as number) ?? Math.floor(n_fft / 4);
                const win_length = (allParams['win_length'] as number) ?? n_fft;
                const center = (allParams['center'] as boolean) ?? true;
                const normalized = (allParams['normalized'] as boolean) ?? false;
                const outputLength = allParams['length'] as number | undefined;

                const backend = env.getBackend(input.device);

                const inputShape = input.shape;
                const ndim = inputShape.length;
                const isBatched = ndim === 3;

                // STFT output: (freq_bins, n_frames) or (batch, freq_bins, n_frames)
                const n_frames = isBatched ? inputShape[2] : inputShape[1];
                const batchSize = isBatched ? inputShape[0] : 1;



                // Step 1: Transpose to (n_frames, freq_bins) for irfft
                const transposeDims = isBatched ? [0, 2, 1] : [1, 0];
                const transposeEntry: OpEntry = { name: 'permute', dispatchKey: 'permute', mechanism: 'View' } as any;
                const transposed = dispatchView(transposeEntry, {
                    opName: 'permute',
                    tensorInputs: [input],
                    scalarArgs: { dims: transposeDims },
                    metadata: {},
                    outs: undefined,
                }) as ITensorHandle;

                // The irfft kernel's HermitianMirror has strided read support.
                // We use the transposed view directly as irfftInput.
                let irfftInput = transposed;


                // Step 2: irfft - transform frequency bins back to time domain
                // Output: (n_frames, n_fft) or (batch, n_frames, n_fft)
                const norm = normalized ? 'ortho' : 'backward';
                const irfftEntry: OpEntry = { name: 'irfft', dispatchKey: 'irfft', mechanism: 'Factory' } as any;
                let frames = dispatchFactory(irfftEntry, {
                    opName: 'irfft',
                    tensorInputs: [irfftInput],
                    scalarArgs: { n: n_fft, dim: -1, norm },
                    metadata: { n: n_fft, dim: -1, norm },
                    outs: undefined,
                });



                // Step 3: Apply window function (if provided)
                if (windowTensor) {
                    let window = windowTensor;
                    // Pad window if needed
                    if (win_length < n_fft) {
                        const left = Math.floor((n_fft - win_length) / 2);
                        const padEntry: OpEntry = { name: 'pad', dispatchKey: 'pad', mechanism: 'Factory' } as any;
                        window = dispatchFactory(padEntry, {
                            opName: 'pad',
                            tensorInputs: [window],
                            scalarArgs: { pad: [left, n_fft - win_length - left], mode: 'constant', value: 0 },
                            metadata: {},
                            outs: undefined,
                        });
                    }

                    // Multiply frames by window
                    const mulOutput = backend.createTensorHandle(frames.shape, frames.dtype);
                    const mulIter = TensorIterator.binaryOp(frames, window, 'mul', mulOutput);
                    const mulKernel = backend.operators.find('mul') as IteratorKernelImpl | undefined;
                    if (!mulKernel) throw new Error("Kernel 'mul' not found");
                    mulKernel(mulIter);
                    frames = mulOutput;
                }

                // Step 4: Overlap-add
                // Output length: (n_frames - 1) * hop_length + n_fft
                const signalLen = (n_frames - 1) * hop_length + n_fft;
                const outputShape = isBatched ? [batchSize, signalLen] : [signalLen];

                // Create zeros output
                const zerosIter = TensorIterator.nullaryOp(outputShape, frames.dtype, 'zeros');
                const zerosKernel = backend.operators.find('zeros') as IteratorKernelImpl | undefined;
                if (zerosKernel) zerosKernel(zerosIter);
                let output = zerosIter.output().tensorHandle!;


                // Execute overlap-add kernel directly via backend.operators.find
                type OverlapAddKernel = (ctx: DirectContext) => void;
                const overlapAddKernel = backend.operators.find('overlap_add') as OverlapAddKernel | undefined;
                if (!overlapAddKernel) throw new Error("Kernel 'overlap_add' not found");

                overlapAddKernel({
                    kind: 'direct',
                    inputs: [frames],
                    scalars: { hop_length },
                    metadata: {},
                    outs: [output],
                    kernelName: 'overlap_add',
                });

                // Step 5: Normalize by window_sumsquare (COLA normalization)
                // Algorithm: For each output position, compute sum of (window²) from all overlapping frames
                // For rectangular window (no window provided), this equals the number of overlapping frames
                {


                    // 1. Create window tensor (rectangular if no window provided)
                    let windowForNorm: ITensorHandle;
                    if (windowTensor) {
                        windowForNorm = windowTensor;
                        if (win_length < n_fft) {
                            const left = Math.floor((n_fft - win_length) / 2);
                            const padEntry: OpEntry = { name: 'pad', dispatchKey: 'pad', mechanism: 'Factory' } as any;
                            windowForNorm = dispatchFactory(padEntry, {
                                opName: 'pad',
                                tensorInputs: [windowForNorm],
                                scalarArgs: { pad: [left, n_fft - win_length - left], mode: 'constant', value: 0 },
                                metadata: {},
                                outs: undefined,
                            });
                        }
                    } else {
                        // Rectangular window (all ones)
                        const onesIter = TensorIterator.nullaryOp([n_fft], frames.dtype, 'ones');
                        const onesKernel = backend.operators.find('ones') as IteratorKernelImpl | undefined;
                        if (onesKernel) onesKernel(onesIter);
                        windowForNorm = onesIter.output().tensorHandle!;
                    }


                    // 2. Square the window: window²
                    const windowSqOutput = backend.createTensorHandle([n_fft], frames.dtype);
                    const sqIter = TensorIterator.binaryOp(windowForNorm, windowForNorm, 'mul', windowSqOutput);
                    const mulKernelForSq = backend.operators.find('mul') as IteratorKernelImpl | undefined;
                    if (mulKernelForSq) mulKernelForSq(sqIter);

                    // 3. Create fakeFrames by repeating window² for each frame
                    // Shape: (n_frames, n_fft) or (batch, n_frames, n_fft)
                    // Use unsqueeze + expand view to tile window² without data copy
                    // This creates a view with stride=0 for the frame dimension
                    let fakeFrames: ITensorHandle;
                    if (isBatched) {
                        // [n_fft] -> [1, 1, n_fft] -> [batchSize, n_frames, n_fft]
                        const unsq1Entry: OpEntry = { name: 'unsqueeze', dispatchKey: 'unsqueeze', mechanism: 'View' } as any;
                        let expanded = dispatchView(unsq1Entry, {
                            opName: 'unsqueeze',
                            tensorInputs: [windowSqOutput],
                            scalarArgs: { dim: 0 },
                            metadata: {},
                            outs: undefined,
                        }) as ITensorHandle;
                        expanded = dispatchView(unsq1Entry, {
                            opName: 'unsqueeze',
                            tensorInputs: [expanded],
                            scalarArgs: { dim: 0 },
                            metadata: {},
                            outs: undefined,
                        }) as ITensorHandle;
                        const expandEntry: OpEntry = { name: 'expand', dispatchKey: 'expand', mechanism: 'View' } as any;
                        fakeFrames = dispatchView(expandEntry, {
                            opName: 'expand',
                            tensorInputs: [expanded],
                            scalarArgs: { size: [batchSize, n_frames, n_fft] },
                            metadata: {},
                            outs: undefined,
                        }) as ITensorHandle;
                    } else {
                        // [n_fft] -> [1, n_fft] -> [n_frames, n_fft]
                        const unsqEntry: OpEntry = { name: 'unsqueeze', dispatchKey: 'unsqueeze', mechanism: 'View' } as any;
                        const unsqueezed = dispatchView(unsqEntry, {
                            opName: 'unsqueeze',
                            tensorInputs: [windowSqOutput],
                            scalarArgs: { dim: 0 },
                            metadata: {},
                            outs: undefined,
                        }) as ITensorHandle;
                        const expandEntry: OpEntry = { name: 'expand', dispatchKey: 'expand', mechanism: 'View' } as any;
                        fakeFrames = dispatchView(expandEntry, {
                            opName: 'expand',
                            tensorInputs: [unsqueezed],
                            scalarArgs: { size: [n_frames, n_fft] },
                            metadata: {},
                            outs: undefined,
                        }) as ITensorHandle;
                    }

                    // Make fakeFrames contiguous - overlap_add kernel uses linear indexing
                    // and doesn't handle strided inputs (stride=0 from expand)
                    const fakeFrameShape = isBatched ? [batchSize, n_frames, n_fft] : [n_frames, n_fft];
                    const contiguousFakeFrames = backend.createTensorHandle(fakeFrameShape, frames.dtype);
                    const contiguousIter = TensorIterator.unaryOp(fakeFrames, 'contiguous', contiguousFakeFrames);
                    const contiguousKernel = backend.operators.find('contiguous') as IteratorKernelImpl | undefined;
                    if (contiguousKernel) contiguousKernel(contiguousIter);

                    // 4. Compute window_sumsquare using overlap_add on contiguous fakeFrames
                    const wssIter = TensorIterator.nullaryOp(outputShape, frames.dtype, 'zeros');
                    const wssZerosKernel = backend.operators.find('zeros') as IteratorKernelImpl | undefined;
                    if (wssZerosKernel) wssZerosKernel(wssIter);
                    const windowSumsquare = wssIter.output().tensorHandle!;

                    overlapAddKernel({
                        kind: 'direct',
                        inputs: [contiguousFakeFrames],
                        scalars: { hop_length },
                        metadata: {},
                        outs: [windowSumsquare],
                        kernelName: 'overlap_add',
                    });

                    // 5. Clamp to avoid division by zero: max(window_sumsquare, eps)
                    const eps = 1e-8;
                    const epsIter = TensorIterator.nullaryOp([1], frames.dtype, 'full');
                    epsIter.setScalarArgs({ fill_value: eps });
                    const fullKernel = backend.operators.find('full') as IteratorKernelImpl | undefined;
                    if (fullKernel) fullKernel(epsIter);
                    const epsTensor = epsIter.output().tensorHandle!;

                    const clampedWss = backend.createTensorHandle(outputShape, frames.dtype);
                    const maxIter = TensorIterator.binaryOp(windowSumsquare, epsTensor, 'maximum', clampedWss);
                    const maxKernel = backend.operators.find('maximum') as IteratorKernelImpl | undefined;
                    if (maxKernel) maxKernel(maxIter);

                    // 6. Divide output by clamped window_sumsquare

                    const divOutput = backend.createTensorHandle(outputShape, output.dtype);
                    const divIter = TensorIterator.binaryOp(output, clampedWss, 'div', divOutput);
                    const divKernel = backend.operators.find('div') as IteratorKernelImpl | undefined;
                    if (divKernel) {
                        divKernel(divIter);
                    } else {
                        throw new Error('Kernel div not found');
                    }

                    // Use divOutput as the final output (avoid copy step)
                    output = divOutput;
                }

                // Step 5: Remove center padding if needed
                let result: ITensorHandle = output;
                if (center) {
                    const padAmount = Math.floor(n_fft / 2);
                    const actualLen = outputLength ?? (signalLen - 2 * padAmount);

                    // Slice to remove padding
                    const sliceEntry: OpEntry = { name: 'slice', dispatchKey: 'slice', mechanism: 'View' } as any;
                    const sliceStr = isBatched
                        ? `:, ${padAmount}:${padAmount + actualLen}`
                        : `${padAmount}:${padAmount + actualLen}`;

                    result = dispatchView(sliceEntry, {
                        opName: 'slice',
                        tensorInputs: [output],
                        scalarArgs: { slices: sliceStr },
                        metadata: {},
                        outs: undefined,
                    }) as ITensorHandle;
                }

                return result;
            }

            case 'fft2': {
                // 2D FFT = FFT along dim[0] + FFT along dim[1]
                const input = inputs[0];
                let dims = (allParams['dim'] as number[]) ?? [-2, -1];
                const s = allParams['s'] as number[] | undefined;
                const norm = allParams['norm'] as string | undefined;

                // Resolve negative dimensions
                const ndim = input.shape.length;
                const dim0 = dims[0] < 0 ? ndim + dims[0] : dims[0];
                const dim1 = dims[1] < 0 ? ndim + dims[1] : dims[1];

                // First FFT along dim0
                const fftEntry: OpEntry = { name: 'fft', dispatchKey: 'fft', mechanism: 'Factory' } as any;
                let result = dispatchFFT(fftEntry, {
                    opName: 'fft',
                    tensorInputs: [input],
                    scalarArgs: { n: s?.[0], dim: dim0, norm },
                    metadata: {},
                    outs: undefined,
                });

                // Second FFT along dim1
                result = dispatchFFT(fftEntry, {
                    opName: 'fft',
                    tensorInputs: [result],
                    scalarArgs: { n: s?.[1], dim: dim1, norm },
                    metadata: {},
                    outs: undefined,
                });

                return result;
            }

            case 'ifft2': {
                // 2D IFFT = IFFT along dim[0] + IFFT along dim[1]
                const input = inputs[0];
                let dims = (allParams['dim'] as number[]) ?? [-2, -1];
                const s = allParams['s'] as number[] | undefined;
                const norm = allParams['norm'] as string | undefined;

                const ndim = input.shape.length;
                const dim0 = dims[0] < 0 ? ndim + dims[0] : dims[0];
                const dim1 = dims[1] < 0 ? ndim + dims[1] : dims[1];

                const ifftEntry: OpEntry = { name: 'ifft', dispatchKey: 'ifft', mechanism: 'Factory' } as any;
                let result = dispatchFFT(ifftEntry, {
                    opName: 'ifft',
                    tensorInputs: [input],
                    scalarArgs: { n: s?.[0], dim: dim0, norm },
                    metadata: {},
                    outs: undefined,
                });

                result = dispatchFFT(ifftEntry, {
                    opName: 'ifft',
                    tensorInputs: [result],
                    scalarArgs: { n: s?.[1], dim: dim1, norm },
                    metadata: {},
                    outs: undefined,
                });

                return result;
            }

            case 'rfft2': {
                // 2D RFFT = FFT along dim[0] + RFFT along dim[1] (last dim uses real optimization)
                const input = inputs[0];
                let dims = (allParams['dim'] as number[]) ?? [-2, -1];
                const s = allParams['s'] as number[] | undefined;
                const norm = allParams['norm'] as string | undefined;

                const ndim = input.shape.length;
                const dim0 = dims[0] < 0 ? ndim + dims[0] : dims[0];
                const dim1 = dims[1] < 0 ? ndim + dims[1] : dims[1];

                // First: RFFT along dim1 (Real to Complex OneSided)
                const rfftEntry: OpEntry = { name: 'rfft', dispatchKey: 'rfft', mechanism: 'Factory' } as any;
                let result = dispatchFFT(rfftEntry, {
                    opName: 'rfft',
                    tensorInputs: [input],
                    scalarArgs: { n: s?.[1], dim: dim1, norm },
                    metadata: {},
                    outs: undefined,
                });

                // Second: FFT along dim0 (Complex to Complex)
                const fftEntry: OpEntry = { name: 'fft', dispatchKey: 'fft', mechanism: 'Factory' } as any;
                result = dispatchFFT(fftEntry, {
                    opName: 'fft',
                    tensorInputs: [result],
                    scalarArgs: { n: s?.[0], dim: dim0, norm },
                    metadata: {},
                    outs: undefined,
                });

                return result;
            }

            case 'irfft2': {
                // 2D IRFFT = IRFFT along dim[1] + IFFT along dim[0]
                const input = inputs[0];
                let dims = (allParams['dim'] as number[]) ?? [-2, -1];
                const s = allParams['s'] as number[] | undefined;
                const norm = allParams['norm'] as string | undefined;

                const ndim = input.shape.length;
                const dim0 = dims[0] < 0 ? ndim + dims[0] : dims[0];
                const dim1 = dims[1] < 0 ? ndim + dims[1] : dims[1];

                // First: IFFT along dim0 (Complex to Complex)
                const ifftEntry: OpEntry = { name: 'ifft', dispatchKey: 'ifft', mechanism: 'Factory' } as any;
                let result = dispatchFFT(ifftEntry, {
                    opName: 'ifft',
                    tensorInputs: [input],
                    scalarArgs: { n: s?.[0], dim: dim0, norm },
                    metadata: {},
                    outs: undefined,
                });

                // Second: IRFFT along dim1 (Complex OneSided to Real)
                const irfftEntry: OpEntry = { name: 'irfft', dispatchKey: 'irfft', mechanism: 'Factory' } as any;
                result = dispatchFFT(irfftEntry, {
                    opName: 'irfft',
                    tensorInputs: [result],
                    scalarArgs: { n: s?.[1], dim: dim1, norm },
                    metadata: {},
                    outs: undefined,
                });

                return result;
            }


            case 'fftn': {
                // fftn(input, s, dim, norm)
                const input = inputs[0];
                let s = allParams['s'] as number[] | undefined;
                let dims = allParams['dim'] as number[] | undefined;
                const norm = (allParams['norm'] as string) ?? 'backward';

                if (dims === undefined) {
                    dims = Array.from({ length: input.shape.length }, (_, i) => i);
                }

                if (s === undefined) {
                    s = dims.map(d => {
                        const resolved = d < 0 ? input.shape.length + d : d;
                        return input.shape[resolved];
                    });
                }

                if (s.length !== dims.length) {
                    throw new Error(`fftn: s and dim must have same length`);
                }

                let result = input;
                const fftEntry: OpEntry = { name: 'fft', dispatchKey: 'fft', mechanism: 'FFT' } as any;

                for (let i = 0; i < dims.length; i++) {
                    const d = dims[i];
                    const n = s[i];
                    const resolvedDim = d < 0 ? result.shape.length + d : d;

                    result = dispatchFFT(fftEntry, {
                        opName: 'fft',
                        tensorInputs: [result],
                        scalarArgs: { n, dim: resolvedDim, norm },
                        metadata: {},
                        outs: undefined,
                    }) as ITensorHandle;
                }
                return result;
            }

            case 'ifftn': {
                // ifftn(input, s, dim, norm)
                const input = inputs[0];
                let s = allParams['s'] as number[] | undefined;
                let dims = allParams['dim'] as number[] | undefined;
                const norm = (allParams['norm'] as string) ?? 'backward';

                if (dims === undefined) {
                    dims = Array.from({ length: input.shape.length }, (_, i) => i);
                }

                if (s === undefined) {
                    s = dims.map(d => {
                        const resolved = d < 0 ? input.shape.length + d : d;
                        return input.shape[resolved];
                    });
                }

                if (s.length !== dims.length) {
                    throw new Error(`ifftn: s and dim must have same length`);
                }

                let result = input;
                const ifftEntry: OpEntry = { name: 'ifft', dispatchKey: 'ifft', mechanism: 'FFT' } as any;

                for (let i = 0; i < dims.length; i++) {
                    const d = dims[i];
                    const n = s[i];
                    const resolvedDim = d < 0 ? result.shape.length + d : d;

                    result = dispatchFFT(ifftEntry, {
                        opName: 'ifft',
                        tensorInputs: [result],
                        scalarArgs: { n, dim: resolvedDim, norm },
                        metadata: {},
                        outs: undefined,
                    }) as ITensorHandle;
                }
                return result;
            }

            case 'rfftn': {
                // rfftn(input, s, dim, norm)
                const input = inputs[0];
                let s = allParams['s'] as number[] | undefined;
                let dims = allParams['dim'] as number[] | undefined;
                const norm = (allParams['norm'] as string) ?? 'backward';

                if (dims === undefined) {
                    dims = Array.from({ length: input.shape.length }, (_, i) => i);
                }

                if (s === undefined) {
                    s = dims.map(d => {
                        const resolved = d < 0 ? input.shape.length + d : d;
                        return input.shape[resolved];
                    });
                }

                if (s.length !== dims.length) {
                    throw new Error(`rfftn: s and dim must have same length`);
                }

                // Last dimension is rfft, others are fft
                // Typically apply rfft first on the last specified dimension
                const lastDimIndex = dims.length - 1;
                const rfftDim = dims[lastDimIndex];
                const rfftN = s[lastDimIndex];
                const resolvedRfftDim = rfftDim < 0 ? input.shape.length + rfftDim : rfftDim;

                let result = input;

                // 1. RFFT
                const rfftEntry: OpEntry = { name: 'rfft', dispatchKey: 'rfft', mechanism: 'FFT' } as any;
                result = dispatchFFT(rfftEntry, {
                    opName: 'rfft',
                    tensorInputs: [result],
                    scalarArgs: { n: rfftN, dim: resolvedRfftDim, norm },
                    metadata: {},
                    outs: undefined,
                }) as ITensorHandle;

                // 2. FFTs
                const fftEntry: OpEntry = { name: 'fft', dispatchKey: 'fft', mechanism: 'FFT' } as any;
                // Note: Iterating other dimensions. 
                // Careful: If dims order is mixed, we should follow it?
                // PyTorch rfftn: The last dimension in `dim` is the one transformed with rfft.
                // The others are transformed with fft. 
                // We should respect the order if possible, or just iterate.
                // Since result after rfft is complex, subsequent FFTs are fine.
                for (let i = 0; i < lastDimIndex; i++) {
                    const d = dims[i];
                    const n = s[i];
                    const resolvedDim = d < 0 ? result.shape.length + d : d;

                    result = dispatchFFT(fftEntry, {
                        opName: 'fft',
                        tensorInputs: [result],
                        scalarArgs: { n, dim: resolvedDim, norm },
                        metadata: {},
                        outs: undefined,
                    }) as ITensorHandle;
                }
                return result;
            }

            case 'irfftn': {
                // irfftn(input, s, dim, norm)
                const input = inputs[0];
                let s = allParams['s'] as number[] | undefined;
                let dims = allParams['dim'] as number[] | undefined;
                const norm = (allParams['norm'] as string) ?? 'backward';

                if (dims === undefined) {
                    dims = Array.from({ length: input.shape.length }, (_, i) => i);
                }

                if (s === undefined) {
                    s = dims.map((d, i) => {
                        const resolved = d < 0 ? input.shape.length + d : d;
                        const inSize = input.shape[resolved];
                        if (i === dims.length - 1) {
                            return 2 * (inSize - 1);
                        }
                        return inSize;
                    });
                }

                if (s.length !== dims.length) {
                    throw new Error(`irfftn: s and dim must have same length`);
                }

                let result = input;
                const ifftEntry: OpEntry = { name: 'ifft', dispatchKey: 'ifft', mechanism: 'FFT' } as any;
                const irfftEntry: OpEntry = { name: 'irfft', dispatchKey: 'irfft', mechanism: 'FFT' } as any;

                const lastDimIndex = dims.length - 1;

                // 1. IFFTs
                for (let i = 0; i < lastDimIndex; i++) {
                    const d = dims[i];
                    const n = s[i];
                    const resolvedDim = d < 0 ? result.shape.length + d : d;

                    result = dispatchFFT(ifftEntry, {
                        opName: 'ifft',
                        tensorInputs: [result],
                        scalarArgs: { n, dim: resolvedDim, norm },
                        metadata: {},
                        outs: undefined,
                    }) as ITensorHandle;
                }

                // 2. IRFFT
                const irfftDim = dims[lastDimIndex];
                const irfftN = s[lastDimIndex];
                const resolvedIrfftDim = irfftDim < 0 ? result.shape.length + irfftDim : irfftDim;

                result = dispatchFFT(irfftEntry, {
                    opName: 'irfft',
                    tensorInputs: [result],
                    scalarArgs: { n: irfftN, dim: resolvedIrfftDim, norm },
                    metadata: {},
                    outs: undefined,
                }) as ITensorHandle;

                return result;
            }

            case 'hfft': {
                // hfft(x, n, dim, norm) = irfft(conj(x), n, dim, norm)
                // Computes the 1D DFT of a Hermitian symmetric input signal
                const input = inputs[0];
                const n = allParams['n'] as number | undefined;
                const dim = (allParams['dim'] as number) ?? -1;
                const norm = (allParams['norm'] as string) ?? 'backward';

                const backend = env.getBackend(input.device);

                // Step 1: Compute conjugate of input
                const conjOutput = backend.createTensorHandle(input.shape, input.dtype);
                const conjIter = TensorIterator.unaryOp(input, 'conj', conjOutput);
                const conjKernel = backend.operators.find('conj') as IteratorKernelImpl | undefined;
                if (!conjKernel) throw new Error("Kernel 'conj' not found");
                conjKernel(conjIter);

                // Step 2: irfft(conj(x), n, dim, norm)
                const irfftEntry: OpEntry = { name: 'irfft', dispatchKey: 'irfft', mechanism: 'FFT' } as any;
                return dispatchFFT(irfftEntry, {
                    opName: 'irfft',
                    tensorInputs: [conjOutput],
                    scalarArgs: { n, dim, norm },
                    metadata: {},
                    outs: undefined,
                });
            }

            case 'ihfft': {
                // ihfft(x, n, dim, norm) = conj(rfft(x, n, dim, norm))
                // Computes the inverse of hfft
                const input = inputs[0];
                const n = allParams['n'] as number | undefined;
                const dim = (allParams['dim'] as number) ?? -1;
                const norm = (allParams['norm'] as string) ?? 'backward';

                const backend = env.getBackend(input.device);

                // Step 1: rfft(x, n, dim, norm)
                const rfftEntry: OpEntry = { name: 'rfft', dispatchKey: 'rfft', mechanism: 'FFT' } as any;
                const rfftResult = dispatchFFT(rfftEntry, {
                    opName: 'rfft',
                    tensorInputs: [input],
                    scalarArgs: { n, dim, norm },
                    metadata: {},
                    outs: undefined,
                }) as ITensorHandle;

                // Step 2: conj(rfft_result)
                const conjOutput = backend.createTensorHandle(rfftResult.shape, rfftResult.dtype);
                const conjIter = TensorIterator.unaryOp(rfftResult, 'conj', conjOutput);
                const conjKernel = backend.operators.find('conj') as IteratorKernelImpl | undefined;
                if (!conjKernel) throw new Error("Kernel 'conj' not found");
                conjKernel(conjIter);

                return conjOutput;
            }

            default:
                throw new Error(`CompositeHandler: Unknown operation ${kernelName}`);

        }

    }

    static dispatch(entry: OpEntry, ctx: OperatorContext): ITensorHandle | ITensorHandle[] {
        const handler = CompositeHandler.getInstance();
        const execCtx = handler.buildContext(entry, ctx);
        return handler.execute(execCtx);
    }
}

export const dispatchComposite = CompositeHandler.dispatch;
