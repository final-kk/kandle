/**
 * FFT Kernel Executor
 * 
 * Implements Cooley-Tukey radix-2 FFT using WebGPU multi-pass compute.
 * 
 * Algorithm:
 * 1. Bit-reversal permutation (first pass)
 * 2. log2(N) butterfly stages (iterative passes)
 * 3. Normalization scaling (final pass, if needed)
 */

import type { FFTKernelArgs, RFFTKernelArgs, IRFFTKernelArgs } from './types';
import { buildBitReverseShader } from './shaders/bitReverse.wgsl';
import { buildButterflyShader } from './shaders/butterfly.wgsl';
import { buildScaleShader } from './shaders/scale.wgsl';
import { buildScatterShader } from './shaders/scatter.wgsl';
import { getTwiddleBuffer } from './twiddle';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { createUniformBuffer } from '../../base/uniformUtils';
import { getGlobalDTypeResolver } from '../../base/DTypeResolver';
import { isComplexDtype } from '@kandle/utils';
import type { WebGPUTensor } from '../../base/tensor';
import type { DType, ITensorHandle } from '@kandle/types';
import { executeBluesteinFFT, executeBluesteinRFFT, executeBluesteinIRFFT } from './bluestein';

/**
 * Execute FFT kernel
 */
export function executeFFT(args: FFTKernelArgs): void {
    const { input, output, dim, n, norm, direction, isRealInput } = args;

    // Cast to WebGPU tensors
    const inputTensor = input as WebGPUTensor<DType>;
    const outputTensor = output as WebGPUTensor<DType>;

    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;

    // Validate FFT size is power of 2, otherwise use Bluestein
    const log2N = Math.log2(n);
    if (!Number.isInteger(log2N)) {
        // Non-power-of-2: Use Bluestein algorithm
        return executeBluesteinFFT(args);
    }

    // Calculate batch parameters
    // For a tensor of shape [..., n] with FFT on last dim,
    // batch_size = product of all dims except the FFT dim
    const inputShape = input.shape;
    const outputShape = output.shape;

    // Compute batch size (all dimensions except FFT dim)
    let batchSize = 1;
    for (let i = 0; i < inputShape.length; i++) {
        if (i !== dim) {
            batchSize *= inputShape[i];
        }
    }

    // Get input strides
    const inputStrides = inputTensor.strides;
    const outputStrides = outputTensor.strides;

    // Calculate stride parameters for input and output
    const inputStrideInfo = computeFFTStrides(inputShape, inputStrides, dim);
    const outputStrideInfo = computeFFTStrides(outputShape, outputStrides, dim);

    // Get GPU buffers
    const inputBuffer = inputTensor.storage.buffer as GPUBuffer;
    const outputBuffer = outputTensor.storage.buffer as GPUBuffer;

    // Get byte offsets
    const resolver = getGlobalDTypeResolver();
    const inputBytesPerElement = resolver.getDescriptor(input.dtype).gpuBytesPerElement;
    const outputBytesPerElement = resolver.getDescriptor(output.dtype).gpuBytesPerElement;

    const inputByteOffset = inputTensor.offset * inputBytesPerElement;
    const outputByteOffset = outputTensor.offset * outputBytesPerElement;

    // Use a temporary buffer for intermediate results to handle strided I/O correctly
    // Intermediate layout is always contiguous [Batch, N]
    const tempBufferSize = n * batchSize * 8; // 8 bytes per complex64
    const tempBuffer = device.createBuffer({
        size: tempBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // === Pass 1: Bit-reversal permutation (Input -> Temp) ===
    // Reads from strided input, writes to contiguous temp
    executeBitReversePass(
        device,
        inputBuffer,
        tempBuffer,
        inputByteOffset,
        0, // Temp has 0 offset
        n,
        log2N,
        batchSize,
        inputStrideInfo.strideOuter,
        inputStrideInfo.strideInner,
        inputStrideInfo.fftStride,
        inputStrideInfo.batchInner,
        isRealInput,
        workgroupSize
    );

    // === Pass 2 to log2(N)+1: Butterfly stages (In-place on Temp) ===
    const directionValue = direction === 'forward' ? -1.0 : 1.0;
    const twiddleBuffer = getTwiddleBuffer(device, n);

    for (let stage = 0; stage < log2N; stage++) {
        executeButterflyPass(
            device,
            tempBuffer,
            twiddleBuffer,
            0,
            n,
            stage,
            directionValue,
            batchSize,
            workgroupSize
        );
    }

    // === Final Pass: Normalization (In-place on Temp) ===
    const scale = computeNormalizationScale(n, norm, direction);
    if (scale !== 1.0) {
        executeScalePass(
            device,
            tempBuffer,
            0,
            n * batchSize,
            scale,
            workgroupSize
        );
    }

    // === Scatter Pass: Copy Temp to Output (Temp -> Strided Output) ===
    executeScatterPass(
        device,
        tempBuffer,
        outputBuffer,
        0,
        outputByteOffset,
        n,
        batchSize,
        outputStrideInfo.strideOuter,
        outputStrideInfo.strideInner,
        outputStrideInfo.fftStride,
        outputStrideInfo.batchInner,
        2, // complex64 = 2 elements per pixel
        workgroupSize
    );

    tempBuffer.destroy();
}

/**
 * Compute strides for FFT operation on arbitrary dimension
 * 
 * For FFT on dimension `dim`:
 * - fftStride: stride to move along the FFT axis (strides[dim])
 * - outerStride: stride to move to the next "batch" (product of inner dims up to FFT dim)
 * 
 * The kernel iterates: batch_idx * outerStride + fft_idx * fftStride
 * This allows FFT on any dimension, not just the last one.
 * 
 * NOTE: For non-contiguous batches (e.g., transposed outer dims), this simple model
 * may not work. We assume batch dimensions are effectively flattened and contiguous
 * relative to the FFT dimension.
 */

interface FFTStrideInfo {
    fftStride: number;    // Stride along the FFT dimension
    strideOuter: number;  // Stride of dimensions BEFORE fft dim
    strideInner: number;  // Stride of dimensions AFTER fft dim
    batchInner: number;   // Size of inner batch dimensions (product of dims after fft dim)
    batchSize: number;    // Total number of FFT batches (product of all non-fft dims)
}

function computeFFTStrides(
    shape: readonly number[],
    strides: readonly number[],
    dim: number
): FFTStrideInfo {
    const ndim = shape.length;
    const resolvedDim = dim < 0 ? ndim + dim : dim;

    if (resolvedDim < 0 || resolvedDim >= ndim) {
        throw new Error(`Invalid dimension ${dim} for tensor with ${ndim} dimensions`);
    }

    const fftStride = strides[resolvedDim];

    // Calculate strides and sizes for split batch
    // batch_idx = idx_outer * batchInner + idx_inner

    let batchInner = 1;
    let strideInner = 1; // Default if no inner dims (contiguous if dim is last)

    // Check if there are inner dimensions (to the right of FFT dim)
    if (resolvedDim < ndim - 1) {
        // Product of sizes of inner dims
        for (let i = resolvedDim + 1; i < ndim; i++) {
            batchInner *= shape[i];
        }
        // Stride is stride of the last dimension? No, stride of the inner batch "block"
        // Wait, for inner indices to map linearly, they must be contiguous relative to themselves?
        // Or we use the stride of the *last* dimension as strideInner?
        // Actually, if we use (idx_outer * strideOuter + idx_inner * strideInner),
        // we are assuming a single stride for inner and single for outer.
        // This ONLY works if inner dimensions are contiguous (strideInner=1) or scalar.
        // If inner dimensions are NOT contiguous (e.g. `transpose(0, 1)` on 3D),
        // we might have complex strides.
        // HOWEVER, standard "contiguous" tensors have strides: s[i] = s[i+1]*shape[i+1].
        // So last dim stride is 1.
        // The inner block stride is 1 (element-wise).
        // strideInner should be `strides[ndim-1]` ideally (which is usually 1).

        // Let's rely on the provided strides array.
        // Ideally, `strideInner` is the stride of the last dimension.
        strideInner = strides[ndim - 1];

        // But what if we have multiple inner dimensions?
        // [2, 4, 4], fft on dim 0. Inner dims: [4, 4].
        // Indices 0..15.
        // Inner dimensions layout: i*4 + j.
        // Stride[1]=4, Stride[2]=1.
        // Offset = i*4 + j*1.
        // Can we represent this as `idx * strideInner`?
        // Only if `idx` iterates linearly and access is linear.
        // Yes, if inner block is contiguous, `i*4+j` maps to `idx`.
        // Then `offset` = `idx * 1`. Correct.
        // So `strideInner` = 1 (or stride of last dim).
    } else {
        // Last dimension is FFT. No inner dims.
        batchInner = 1;
        // strideInner won't be used as idx_inner is always 0
        strideInner = 0;
    }

    let strideOuter = 0;
    // Check if there are outer dimensions (to the left of FFT dim)
    if (resolvedDim > 0) {
        // Stride of the dimension just before FFT dim
        strideOuter = strides[resolvedDim - 1];
    } else {
        // First dimension is FFT. idx_outer always 0.
        strideOuter = 0;
    }

    // Calculating total batch size
    let batchSize = 1;
    for (let i = 0; i < ndim; i++) {
        if (i !== resolvedDim) {
            batchSize *= shape[i];
        }
    }

    // Safety fallback for degenerate cases or 1D
    if (ndim === 1) {
        strideOuter = 0;
        strideInner = 0;
        batchInner = 1;
    }

    return { fftStride, strideOuter, strideInner, batchInner, batchSize };
}

// Legacy wrapper  
function getBatchStride(
    shape: readonly number[],
    strides: readonly number[],
    dim: number
): number {
    const info = computeFFTStrides(shape, strides, dim);
    // For legacy calls, we assume last dim behavior mostly?
    // Or we try to return something meaningful.
    return info.strideOuter;
}


/**
 * Compute normalization scale factor
 * @internal Exported for Bluestein algorithm
 */
export function computeNormalizationScale(
    n: number,
    norm: 'forward' | 'backward' | 'ortho',
    direction: 'forward' | 'inverse'
): number {
    switch (norm) {
        case 'backward':
            // No norm on forward, 1/N on inverse
            return direction === 'inverse' ? 1 / n : 1;
        case 'forward':
            // 1/N on forward, no norm on inverse
            return direction === 'forward' ? 1 / n : 1;
        case 'ortho':
            // 1/sqrt(N) on both
            return 1 / Math.sqrt(n);
        default:
            return 1;
    }
}

/**
 * Execute bit-reversal permutation pass
 * @internal Exported for Bluestein algorithm
 */
export function executeBitReversePass(
    device: GPUDevice,
    inputBuffer: GPUBuffer,
    outputBuffer: GPUBuffer,
    inputByteOffset: number,
    outputByteOffset: number,
    n: number,
    log2n: number,
    batchSize: number,
    strideOuter: number,
    strideInner: number,
    fftStride: number,
    batchInner: number,
    isRealInput: boolean,
    workgroupSize: number
): void {
    // Build shader
    const shaderCode = buildBitReverseShader({
        workgroupSize,
        isRealInput,
        inputDtype: isRealInput ? 'float32' : 'complex64',
    });

    // Get or create pipeline
    const pipelineKey = `fft_bitReverse_${isRealInput ? 'real' : 'complex'}_wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // Convert byte offset to logical element offset
    // For real: 4 bytes per f32, for complex: 8 bytes per vec2<f32>
    const bytesPerElement = isRealInput ? 4 : 8;
    const baseOffset = inputByteOffset / bytesPerElement;

    // Create uniform buffer (32 bytes to match shader Params struct)
    // struct Params { n, log2n, batch_size, stride_outer, stride_inner, fft_stride, batch_inner, base_offset }
    const uniformData = new ArrayBuffer(32);
    const u32View = new Uint32Array(uniformData);
    u32View[0] = n;
    u32View[1] = log2n;
    u32View[2] = batchSize;
    u32View[3] = strideOuter;
    u32View[4] = strideInner;
    u32View[5] = fftStride;
    u32View[6] = batchInner;
    u32View[7] = baseOffset; // Input tensor base offset (logical elements)

    const uniformBuffer = createUniformBuffer(uniformData);

    // Create bind group - use offset=0 to avoid 256-byte alignment requirement
    // The base_offset uniform handles the logical offset instead
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: inputBuffer, offset: 0 } },
            { binding: 1, resource: { buffer: outputBuffer, offset: outputByteOffset } },
            { binding: 2, resource: { buffer: uniformBuffer } },
        ],
    });

    // Dispatch
    const numWorkgroupsX = Math.ceil(n / workgroupSize);
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroupsX, batchSize);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
}



/**
 * Execute butterfly pass for a single FFT stage
 * @internal Exported for Bluestein algorithm
 */
export function executeButterflyPass(
    device: GPUDevice,
    dataBuffer: GPUBuffer,
    twiddleBuffer: GPUBuffer,
    byteOffset: number,
    n: number,
    stage: number,
    direction: number,
    batchSize: number,
    workgroupSize: number
): void {
    // Build shader
    const shaderCode = buildButterflyShader({ workgroupSize });

    // Get or create pipeline
    const pipelineKey = `fft_butterfly_wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // Create uniform buffer
    // struct Params { n: u32, stage: u32, direction: f32, batch_size: u32 }
    const uniformData = new ArrayBuffer(16);
    const u32View = new Uint32Array(uniformData);
    const f32View = new Float32Array(uniformData);
    u32View[0] = n;
    u32View[1] = stage;
    f32View[2] = direction;
    u32View[3] = batchSize;

    const uniformBuffer = createUniformBuffer(uniformData);

    // Create bind group
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: dataBuffer, offset: byteOffset } },
            { binding: 1, resource: { buffer: uniformBuffer } },
            { binding: 2, resource: { buffer: twiddleBuffer } },
        ],
    });

    // Dispatch - one thread per butterfly pair
    const numButterflies = n >> 1;
    const numWorkgroupsX = Math.ceil(numButterflies / workgroupSize);
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroupsX, batchSize);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
}

/**
 * Execute scaling pass for normalization
 * @internal Exported for Bluestein algorithm  
 */
export function executeScalePass(
    device: GPUDevice,
    dataBuffer: GPUBuffer,
    byteOffset: number,
    totalElements: number,
    scale: number,
    workgroupSize: number
): void {
    // Build shader
    const shaderCode = buildScaleShader({ workgroupSize });

    // Get or create pipeline
    const pipelineKey = `fft_scale_wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // Create uniform buffer
    // struct Params { total_elements: u32, scale: f32, _padding0: u32, _padding1: u32 }
    const uniformData = new ArrayBuffer(16);
    const u32View = new Uint32Array(uniformData);
    const f32View = new Float32Array(uniformData);
    u32View[0] = totalElements;
    f32View[1] = scale;
    u32View[2] = 0; // padding
    u32View[3] = 0; // padding

    const uniformBuffer = createUniformBuffer(uniformData);

    // Create bind group
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: dataBuffer, offset: byteOffset } },
            { binding: 1, resource: { buffer: uniformBuffer } },
        ],
    });

    // Dispatch
    const numWorkgroups = Math.ceil(totalElements / workgroupSize);
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
}



/**
 * Execute scatter pass (copy contiguous to strided)
 */
function executeScatterPass(
    device: GPUDevice,
    srcBuffer: GPUBuffer,
    dstBuffer: GPUBuffer,
    srcByteOffset: number,
    dstByteOffset: number,
    n: number,
    batchSize: number,
    strideOuter: number,
    strideInner: number,
    fftStride: number,
    batchInner: number,
    elementsPerPixel: number,
    workgroupSize: number
): void {
    const shaderCode = buildScatterShader({ workgroupSize, elementsPerPixel });

    const pipelineKey = `fft_scatter_${elementsPerPixel}_wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    const uniformData = new ArrayBuffer(32);
    const u32View = new Uint32Array(uniformData);
    u32View[0] = n;
    u32View[1] = batchSize;
    u32View[2] = strideOuter;
    u32View[3] = strideInner;
    u32View[4] = fftStride;
    u32View[5] = batchInner;
    u32View[6] = 0;
    u32View[7] = 0;

    const uniformBuffer = createUniformBuffer(uniformData);

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: srcBuffer, offset: srcByteOffset } },
            { binding: 1, resource: { buffer: dstBuffer, offset: dstByteOffset } },
            { binding: 2, resource: { buffer: uniformBuffer } },
        ],
    });

    const numWorkgroupsX = Math.ceil(n / workgroupSize);
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroupsX, batchSize);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
}

/**
 * Execute RFFT kernel (Real to Complex, onesided)
 * 
 * Algorithm:
 * 1. Execute full FFT on real input (treated as complex with imag=0)
 * 2. Output only the first n//2 + 1 frequency bins (utilizing Hermitian symmetry)
 */
export function executeRFFT(args: RFFTKernelArgs): void {
    const { input, output, dim, n, norm, onesidedLen } = args;

    // Cast to WebGPU tensors
    const inputTensor = input as WebGPUTensor<DType>;
    const outputTensor = output as WebGPUTensor<DType>;

    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;

    // Validate FFT size is power of 2, otherwise use Bluestein
    const log2N = Math.log2(n);
    if (!Number.isInteger(log2N)) {
        // Non-power-of-2: Use Bluestein algorithm for RFFT
        return executeBluesteinRFFT(args);
    }

    // Calculate batch parameters
    const inputShape = input.shape;
    let batchSize = 1;
    for (let i = 0; i < inputShape.length; i++) {
        if (i !== dim) {
            batchSize *= inputShape[i];
        }
    }

    const inputStrides = inputTensor.strides;
    const outputStrides = outputTensor.strides;

    // Calculate input strides (RFFT input is real, potentially strided)
    const inputStrideInfo = computeFFTStrides(inputShape, inputStrides, dim);

    // Calculate output strides for scatter (RFFT output is complex, strided)
    const outputStrideInfo = computeFFTStrides(output.shape, outputStrides, dim);

    // Get GPU buffers
    const inputBuffer = inputTensor.storage.buffer as GPUBuffer;
    const outputBuffer = outputTensor.storage.buffer as GPUBuffer;

    // Get byte offsets
    const resolver = getGlobalDTypeResolver();
    const inputBytesPerElement = resolver.getDescriptor(input.dtype).gpuBytesPerElement;
    const outputBytesPerElement = resolver.getDescriptor(output.dtype).gpuBytesPerElement;

    const inputByteOffset = inputTensor.offset * inputBytesPerElement;
    const outputByteOffset = outputTensor.offset * outputBytesPerElement;

    // Create temporary buffer for full FFT result
    const tempBufferSize = n * batchSize * 8; // 8 bytes per complex64
    const tempBuffer = device.createBuffer({
        size: tempBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // === Step 1: Bit-reversal with real-to-complex conversion ===
    executeBitReversePass(
        device,
        inputBuffer,
        tempBuffer,
        inputByteOffset,
        0,
        n,
        log2N,
        batchSize,
        inputStrideInfo.strideOuter,
        inputStrideInfo.strideInner,
        inputStrideInfo.fftStride,
        inputStrideInfo.batchInner,
        true, // isRealInput
        workgroupSize
    );

    // === Step 2: Butterfly stages ===
    const directionValue = -1.0; // forward
    const twiddleBuffer = getTwiddleBuffer(device, n);
    for (let stage = 0; stage < log2N; stage++) {
        executeButterflyPass(
            device,
            tempBuffer,
            twiddleBuffer,
            0,
            n,
            stage,
            directionValue,
            batchSize,
            workgroupSize
        );
    }

    // === Step 3: Normalization ===
    const scale = computeNormalizationScale(n, norm, 'forward');
    if (scale !== 1.0) {
        executeScalePass(device, tempBuffer, 0, n * batchSize, scale, workgroupSize);
    }

    // === Step 4: Copy onesided to output (with strided support) ===
    executeCopyOnesidedPass(
        device,
        tempBuffer,
        outputBuffer,
        0,
        outputByteOffset,
        n,
        onesidedLen,
        batchSize,
        outputStrideInfo.strideOuter,
        outputStrideInfo.strideInner,
        outputStrideInfo.fftStride,
        outputStrideInfo.batchInner,
        workgroupSize
    );

    // Clean up temp buffer (Note: in production, consider buffer pooling)
    tempBuffer.destroy();
}

/**
 * Execute IRFFT kernel (Complex onesided to Real)
 * 
 * Algorithm:
 * 1. Reconstruct full spectrum from onesided input using Hermitian symmetry
 * 2. Execute IFFT
 * 3. Extract real part to output
 */
export function executeIRFFT(args: IRFFTKernelArgs): void {
    const { input, output, dim, n, norm, onesidedLen } = args;

    // Cast to WebGPU tensors
    const inputTensor = input as WebGPUTensor<DType>;
    const outputTensor = output as WebGPUTensor<DType>;

    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;

    // Validate FFT size is power of 2, otherwise use Bluestein
    const log2N = Math.log2(n);
    if (!Number.isInteger(log2N)) {
        // Non-power-of-2: Use Bluestein algorithm for IRFFT
        return executeBluesteinIRFFT(args);
    }

    // Calculate batch parameters
    const inputShape = input.shape;
    let batchSize = 1;
    for (let i = 0; i < inputShape.length; i++) {
        if (i !== dim) {
            batchSize *= inputShape[i];
        }
    }

    // Get GPU buffers
    const inputBuffer = inputTensor.storage.buffer as GPUBuffer;
    const outputBuffer = outputTensor.storage.buffer as GPUBuffer;

    // Get byte offsets
    const resolver = getGlobalDTypeResolver();
    const inputBytesPerElement = resolver.getDescriptor(input.dtype).gpuBytesPerElement;
    const outputBytesPerElement = resolver.getDescriptor(output.dtype).gpuBytesPerElement;

    const inputByteOffset = inputTensor.offset * inputBytesPerElement;
    const outputByteOffset = outputTensor.offset * outputBytesPerElement;

    const outputStrides = outputTensor.strides;
    const outputStrideInfo = computeFFTStrides(output.shape, outputStrides, dim);

    // Create temporary buffer for full spectrum
    const tempBufferSize = n * batchSize * 8; // 8 bytes per complex64
    const tempBuffer = device.createBuffer({
        size: tempBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Get input strides for strided access support
    const inputStrides = inputTensor.strides;
    const inputStrideInfo = computeFFTStrides(inputShape, inputStrides, dim);

    // === Step 1: Copy onesided and mirror to full spectrum ===
    executeHermitianMirrorPass(
        device,
        inputBuffer,
        tempBuffer,
        inputByteOffset,
        0,
        n,
        onesidedLen,
        batchSize,
        inputStrideInfo.strideOuter,
        inputStrideInfo.strideInner,
        inputStrideInfo.fftStride,
        inputStrideInfo.batchInner,
        workgroupSize
    );

    // === Step 2: Bit-reversal permutation (complex to complex) ===
    // Note: We need to do in-place bit-reverse since data is already in tempBuffer
    const tempBuffer2 = device.createBuffer({
        size: tempBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Copy with bit-reversal
    executeBitReversePass(
        device,
        tempBuffer,
        tempBuffer2,
        0,
        0,
        n,
        log2N,
        batchSize,
        n, // strideOuter = n for contiguous complex data
        0, // strideInner = 0 because batchInner is 1 for contiguous [Batch, N]
        1, // fftStride = 1
        1, // batchInner = 1
        false, // isRealInput = false
        workgroupSize
    );

    // === Step 3: Butterfly stages (IFFT direction) ===
    const directionValue = 1.0; // inverse
    const twiddleBuffer = getTwiddleBuffer(device, n);
    for (let stage = 0; stage < log2N; stage++) {
        executeButterflyPass(
            device,
            tempBuffer2,
            twiddleBuffer,
            0,
            n,
            stage,
            directionValue,
            batchSize,
            workgroupSize
        );
    }

    // === Step 4: Normalization ===
    const scale = computeNormalizationScale(n, norm, 'inverse');
    if (scale !== 1.0) {
        executeScalePass(device, tempBuffer2, 0, n * batchSize, scale, workgroupSize);
    }

    // === Step 5: Extract real part to output ===
    executeExtractRealPass(
        device,
        tempBuffer2,
        outputBuffer,
        0,
        outputByteOffset,
        n,
        batchSize,
        outputStrideInfo.strideOuter,
        outputStrideInfo.strideInner,
        outputStrideInfo.fftStride,
        outputStrideInfo.batchInner,
        workgroupSize
    );

    // Clean up temp buffers
    tempBuffer.destroy();
    tempBuffer2.destroy();
}

/**
 * Execute copy onesided pass (copy first n//2+1 elements from full FFT)
 * Supports strided output.
 */
function executeCopyOnesidedPass(
    device: GPUDevice,
    srcBuffer: GPUBuffer,
    dstBuffer: GPUBuffer,
    srcByteOffset: number,
    dstByteOffset: number,
    fullLen: number,
    onesidedLen: number,
    batchSize: number,
    strideOuter: number,
    strideInner: number,
    fftStride: number,
    batchInner: number,
    workgroupSize: number
): void {
    const shaderCode = `
struct Params {
    full_len: u32,
    onesided_len: u32,
    batch_size: u32,
    stride_outer: u32,
    stride_inner: u32,
    fft_stride: u32,
    batch_inner: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.y;
    let idx = gid.x;
    
    if (batch_idx >= params.batch_size || idx >= params.onesided_len) { return; }
    
    // Source: contiguous [Batch, FullLen]
    let src_offset = batch_idx * params.full_len + idx;
    
    // Decode batch index
    let idx_inner = batch_idx % params.batch_inner;
    let idx_outer = batch_idx / params.batch_inner;

    // Dest: strided
    let dst_offset = idx_outer * params.stride_outer + idx_inner * params.stride_inner + idx * params.fft_stride;
    
    dst[dst_offset] = src[src_offset];
}
`;

    const pipelineKey = `fft_copyOnesided_wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    const uniformData = new ArrayBuffer(32);
    const u32View = new Uint32Array(uniformData);
    u32View[0] = fullLen;
    u32View[1] = onesidedLen;
    u32View[2] = batchSize;
    u32View[3] = strideOuter;
    u32View[4] = strideInner;
    u32View[5] = fftStride;
    u32View[6] = batchInner;
    u32View[7] = 0;

    const uniformBuffer = createUniformBuffer(uniformData);

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: srcBuffer, offset: srcByteOffset } },
            { binding: 1, resource: { buffer: dstBuffer, offset: dstByteOffset } },
            { binding: 2, resource: { buffer: uniformBuffer } },
        ],
    });

    const numWorkgroupsX = Math.ceil(onesidedLen / workgroupSize);
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroupsX, batchSize);

    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
}

/**
 * Execute Hermitian mirror pass (reconstruct full spectrum from onesided)
 * X[k] for k = 0..n//2 comes from input
 * X[n-k] = conj(X[k]) for k = 1..n//2-1
 */
function executeHermitianMirrorPass(
    device: GPUDevice,
    srcBuffer: GPUBuffer,
    dstBuffer: GPUBuffer,
    srcByteOffset: number,
    dstByteOffset: number,
    fullLen: number,
    onesidedLen: number,
    batchSize: number,
    strideOuter: number,
    strideInner: number,
    fftStride: number,
    batchInner: number,
    workgroupSize: number
): void {
    // Check if input is contiguous (fast path)
    const isContiguous = (strideOuter === onesidedLen) && (fftStride === 1) && (batchInner === 1);

    // Convert byte offset to logical element offset (8 bytes per vec2<f32>)
    const baseOffset = srcByteOffset / 8;

    const shaderCode = `
// Hermitian mirror with strided input support
struct Params {
    full_len: u32,
    onesided_len: u32,
    batch_size: u32,
    stride_outer: u32,
    stride_inner: u32,
    fft_stride: u32,
    batch_inner: u32,
    base_offset: u32,  // Input tensor base offset (logical elements)
}

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.y;
    let idx = gid.x;
    
    if (batch_idx >= params.batch_size || idx >= params.full_len) { return; }
    
    // Destination is always contiguous
    let dst_offset = batch_idx * params.full_len + idx;
    
    // Decode batch index
    let idx_inner = batch_idx % params.batch_inner;
    let idx_outer = batch_idx / params.batch_inner;

    if (idx < params.onesided_len) {
        // Direct copy for indices 0..onesided_len-1
        // Strided input with base_offset
        let src_offset = params.base_offset + idx_outer * params.stride_outer + idx_inner * params.stride_inner + idx * params.fft_stride;
        dst[dst_offset] = src[src_offset];
    } else {
        // Mirror with conjugate for indices onesided_len..full_len-1
        // X[full_len - k] = conj(X[k])
        let mirror_idx = params.full_len - idx;
        let src_offset = params.base_offset + idx_outer * params.stride_outer + idx_inner * params.stride_inner + mirror_idx * params.fft_stride;
        let val = src[src_offset];
        dst[dst_offset] = vec2<f32>(val.x, -val.y); // conjugate
    }
}
`;

    const pipelineKey = `fft_hermitianMirror_wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // Uniform buffer with base_offset (32 bytes = 8 u32s)
    const uniformData = new ArrayBuffer(32);
    const u32View = new Uint32Array(uniformData);
    u32View[0] = fullLen;
    u32View[1] = onesidedLen;
    u32View[2] = batchSize;
    u32View[3] = strideOuter;
    u32View[4] = strideInner;
    u32View[5] = fftStride;
    u32View[6] = batchInner;
    u32View[7] = baseOffset;

    const uniformBuffer = createUniformBuffer(uniformData);

    // Use offset=0 to avoid 256-byte alignment requirement
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: srcBuffer, offset: 0 } },
            { binding: 1, resource: { buffer: dstBuffer, offset: dstByteOffset } },
            { binding: 2, resource: { buffer: uniformBuffer } },
        ],
    });

    const numWorkgroupsX = Math.ceil(fullLen / workgroupSize);
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroupsX, batchSize);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
}

/**
 * Execute extract real pass (extract real part from complex tensor)
 * Supports strided output.
 */
function executeExtractRealPass(
    device: GPUDevice,
    srcBuffer: GPUBuffer,
    dstBuffer: GPUBuffer,
    srcByteOffset: number,
    dstByteOffset: number,
    len: number,
    batchSize: number,
    strideOuter: number,
    strideInner: number,
    fftStride: number,
    batchInner: number,
    workgroupSize: number
): void {
    const shaderCode = `
struct Params {
    len: u32,
    batch_size: u32,
    stride_outer: u32,
    stride_inner: u32,
    fft_stride: u32,
    batch_inner: u32,
    _padding: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.y;
    let idx = gid.x;
    
    if (batch_idx >= params.batch_size || idx >= params.len) { return; }
    
    // Source: contiguous [Batch, N] (complex)
    let src_offset = batch_idx * params.len + idx;
    
    // Decode batch index
    let idx_inner = batch_idx % params.batch_inner;
    let idx_outer = batch_idx / params.batch_inner;

    // Dest: strided (real)
    let dst_offset = idx_outer * params.stride_outer + idx_inner * params.stride_inner + idx * params.fft_stride;
    
    dst[dst_offset] = src[src_offset].x; // Extract real part
}
`;

    const pipelineKey = `fft_extractReal_wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    const uniformData = new ArrayBuffer(32);
    const u32View = new Uint32Array(uniformData);
    u32View[0] = len;
    u32View[1] = batchSize;
    u32View[2] = strideOuter;
    u32View[3] = strideInner;
    u32View[4] = fftStride;
    u32View[5] = batchInner;
    u32View[6] = 0;
    u32View[7] = 0;

    const uniformBuffer = createUniformBuffer(uniformData);

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: srcBuffer, offset: srcByteOffset } },
            { binding: 1, resource: { buffer: dstBuffer, offset: dstByteOffset } },
            { binding: 2, resource: { buffer: uniformBuffer } },
        ],
    });

    const numWorkgroupsX = Math.ceil(len / workgroupSize);
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroupsX, batchSize);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
}

