/**
 * Bluestein FFT Algorithm (Chirp Z-Transform)
 * 
 * Implements FFT for arbitrary lengths (non-power-of-2) using the
 * Bluestein algorithm, which converts DFT to a convolution problem.
 * 
 * Algorithm:
 * 1. Generate chirp sequence: exp(-πi·n²/N)
 * 2. Modulate input by chirp: a[n] = x[n] * chirp[n]
 * 3. Zero-pad to M >= 2N-1 (power of 2)
 * 4. Compute convolution via FFT: IFFT(FFT(a) * FFT(b))
 * 5. Extract result and demodulate by chirp
 */

import type { FFTKernelArgs, RFFTKernelArgs, IRFFTKernelArgs } from './types';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { createUniformBuffer } from '../../base/uniformUtils';
import { getGlobalDTypeResolver } from '../../base/DTypeResolver';
import type { WebGPUTensor } from '../../base/tensor';
import type { DType } from '@kandle/types';

// Import internal FFT functions
import {
    executeBitReversePass,
    executeButterflyPass,
    executeScalePass,
    computeNormalizationScale,
} from './executor';
import { getTwiddleBuffer } from './twiddle';

/**
 * Find next power of 2 >= n
 */
function nextPowerOf2(n: number): number {
    if (n <= 1) return 1;
    return 1 << Math.ceil(Math.log2(n));
}

/**
 * Generate chirp sequence: exp(-πi·n²/N) for n = 0..len-1
 * Returns interleaved [re, im, re, im, ...] array
 */
function generateChirp(N: number, direction: 'forward' | 'inverse'): Float32Array {
    const sign = direction === 'forward' ? -1 : 1;
    const chirp = new Float32Array(N * 2);

    for (let n = 0; n < N; n++) {
        const angle = sign * Math.PI * n * n / N;
        chirp[n * 2] = Math.cos(angle);
        chirp[n * 2 + 1] = Math.sin(angle);
    }

    return chirp;
}

/**
 * Generate zero-padded chirp convolution kernel
 * b[k] = conj(chirp[-k]) for k = 0..N-1, then zeros, then conj(chirp[-(M-N+k)]) for k = M-N+1..M-1
 */
function generateChirpKernel(N: number, M: number, direction: 'forward' | 'inverse'): Float32Array {
    const sign = direction === 'forward' ? 1 : -1; // Conjugate has opposite sign
    const kernel = new Float32Array(M * 2);

    // Fill b[0..N-1] with conj(chirp[-k]) = conj(chirp[k]) for negatives via symmetry
    // chirp[-k] = exp(-πi·k²/N) for negative indices means exp(-πi·k²/N)
    // conj(chirp[-k]) = exp(+πi·k²/N)

    for (let k = 0; k < N; k++) {
        const angle = sign * Math.PI * k * k / N;
        kernel[k * 2] = Math.cos(angle);
        kernel[k * 2 + 1] = Math.sin(angle);
    }

    // Fill zeros from N to M-N (already zero-initialized)

    // Fill b[M-N+1..M-1] with conj(chirp[-(M-k)]) = conj(chirp[k-M])
    // These are the "wrap-around" values for circular convolution
    for (let k = M - N + 1; k < M; k++) {
        const idx = k - M; // Negative index
        const n = -idx; // Positive
        const angle = sign * Math.PI * n * n / N;
        kernel[k * 2] = Math.cos(angle);
        kernel[k * 2 + 1] = Math.sin(angle);
    }

    return kernel;
}

/**
 * Execute Bluestein FFT for non-power-of-2 sizes
 */
export function executeBluesteinFFT(args: FFTKernelArgs): void {
    const { input, output, dim, n, norm, direction, isRealInput } = args;

    const inputTensor = input as WebGPUTensor<DType>;
    const outputTensor = output as WebGPUTensor<DType>;

    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;

    // Calculate padded size M >= 2N-1, must be power of 2
    const M = nextPowerOf2(2 * n - 1);
    const log2M = Math.log2(M);

    // Calculate batch size
    const inputShape = input.shape;
    let batchSize = 1;
    for (let i = 0; i < inputShape.length; i++) {
        if (i !== dim) {
            batchSize *= inputShape[i];
        }
    }

    // Get GPU buffers and offsets
    const inputBuffer = inputTensor.storage.buffer as GPUBuffer;
    const outputBuffer = outputTensor.storage.buffer as GPUBuffer;

    const resolver = getGlobalDTypeResolver();
    const inputBytesPerElement = resolver.getDescriptor(input.dtype).gpuBytesPerElement;
    const outputBytesPerElement = resolver.getDescriptor(output.dtype).gpuBytesPerElement;

    const inputByteOffset = inputTensor.offset * inputBytesPerElement;
    const outputByteOffset = outputTensor.offset * outputBytesPerElement;

    // === Step 1: Generate chirp sequences on CPU ===
    const chirp = generateChirp(n, direction);
    const chirpKernel = generateChirpKernel(n, M, direction);

    // Upload chirp to GPU
    const chirpBuffer = device.createBuffer({
        size: chirp.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(chirpBuffer, 0, chirp.buffer);

    // === Step 2: Create temporary buffers ===
    // Buffer A: modulated and zero-padded input [batch, M] complex
    const bufferASize = M * batchSize * 8;
    const bufferA = device.createBuffer({
        size: bufferASize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Buffer B: FFT of chirp kernel [M] complex (shared across batches)
    const bufferBSize = M * 8;
    const bufferB = device.createBuffer({
        size: bufferBSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // === Step 3: Prepare chirp kernel FFT (single batch) ===
    const chirpKernelBuffer = device.createBuffer({
        size: chirpKernel.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(chirpKernelBuffer, 0, chirpKernel.buffer);

    // Bit-reverse chirp kernel
    executeBitReversePass(
        device, chirpKernelBuffer, bufferB, 0, 0,
        M, log2M, 1, // batchSize = 1 for kernel
        0, 0, 1, 1,  // strideOuter=0, strideInner=0, fftStride=1, batchInner=1
        false, workgroupSize
    );

    // FFT butterfly on chirp kernel
    const twiddleBufferM = getTwiddleBuffer(device, M);
    for (let stage = 0; stage < log2M; stage++) {
        executeButterflyPass(device, bufferB, twiddleBufferM, 0, M, stage, -1.0, 1, workgroupSize);
    }

    chirpKernelBuffer.destroy();

    // === Step 4: Modulate input and zero-pad to buffer A ===
    executeChirpModulatePass(
        device,
        inputBuffer,
        chirpBuffer,
        bufferA,
        inputByteOffset,
        n,
        M,
        batchSize,
        isRealInput,
        workgroupSize
    );

    // === Step 5: FFT on modulated input ===
    // Need bit-reverse first (in-place by using temp)
    const tempBuffer = device.createBuffer({
        size: bufferASize,
        usage: GPUBufferUsage.STORAGE,
    });

    // Bit-reverse A -> temp
    executeBitReversePass(
        device, bufferA, tempBuffer, 0, 0,
        M, log2M, batchSize,
        M, 0, 1, 1, // Contiguous [batch, M]
        false, workgroupSize
    );

    // Butterfly on temp
    for (let stage = 0; stage < log2M; stage++) {
        executeButterflyPass(device, tempBuffer, twiddleBufferM, 0, M, stage, -1.0, batchSize, workgroupSize);
    }

    // === Step 6: Element-wise multiply temp * bufferB ===
    executeComplexMultiplyPass(
        device,
        tempBuffer,
        bufferB,
        bufferA, // Result back to A
        M,
        batchSize,
        workgroupSize
    );

    // === Step 7: IFFT on result ===
    // Bit-reverse A -> temp
    executeBitReversePass(
        device, bufferA, tempBuffer, 0, 0,
        M, log2M, batchSize,
        M, 0, 1, 1,
        false, workgroupSize
    );

    // Butterfly (inverse direction)
    for (let stage = 0; stage < log2M; stage++) {
        executeButterflyPass(device, tempBuffer, twiddleBufferM, 0, M, stage, 1.0, batchSize, workgroupSize);
    }

    // Scale by 1/M for IFFT normalization
    executeScalePass(device, tempBuffer, 0, M * batchSize, 1.0 / M, workgroupSize);

    // === Step 8: Extract and demodulate ===
    const finalScale = computeNormalizationScale(n, norm, direction);

    executeChirpDemodulatePass(
        device,
        tempBuffer,
        chirpBuffer,
        outputBuffer,
        outputByteOffset,
        n,
        M,
        batchSize,
        finalScale,
        workgroupSize
    );

    // Cleanup
    chirpBuffer.destroy();
    bufferA.destroy();
    bufferB.destroy();
    tempBuffer.destroy();
}

// === Bluestein-specific shader passes ===

/**
 * Modulate input by chirp and zero-pad
 * output[k] = input[k] * chirp[k] for k < n
 * output[k] = 0 for k >= n
 */
function executeChirpModulatePass(
    device: GPUDevice,
    inputBuffer: GPUBuffer,
    chirpBuffer: GPUBuffer,
    outputBuffer: GPUBuffer,
    inputByteOffset: number,
    n: number,
    M: number,
    batchSize: number,
    isRealInput: boolean,
    workgroupSize: number
): void {
    const shaderCode = buildChirpModulateShader({ workgroupSize, isRealInput });

    const pipelineKey = `bluestein_modulate_${isRealInput ? 'real' : 'complex'}_wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    const uniformData = new ArrayBuffer(16);
    const u32View = new Uint32Array(uniformData);
    u32View[0] = n;
    u32View[1] = M;
    u32View[2] = batchSize;
    u32View[3] = 0;

    const uniformBuffer = createUniformBuffer(uniformData);

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: inputBuffer, offset: inputByteOffset } },
            { binding: 1, resource: { buffer: chirpBuffer } },
            { binding: 2, resource: { buffer: outputBuffer } },
            { binding: 3, resource: { buffer: uniformBuffer } },
        ],
    });

    const numWorkgroupsX = Math.ceil(M / workgroupSize);
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroupsX, batchSize);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
}

/**
 * Complex element-wise multiplication
 */
function executeComplexMultiplyPass(
    device: GPUDevice,
    inputA: GPUBuffer,
    inputB: GPUBuffer,
    output: GPUBuffer,
    M: number,
    batchSize: number,
    workgroupSize: number
): void {
    const shaderCode = buildComplexMultiplyShader({ workgroupSize });

    const pipelineKey = `bluestein_cmul_wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    const uniformData = new ArrayBuffer(16);
    const u32View = new Uint32Array(uniformData);
    u32View[0] = M;
    u32View[1] = batchSize;
    u32View[2] = 0;
    u32View[3] = 0;

    const uniformBuffer = createUniformBuffer(uniformData);

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: inputA } },
            { binding: 1, resource: { buffer: inputB } },
            { binding: 2, resource: { buffer: output } },
            { binding: 3, resource: { buffer: uniformBuffer } },
        ],
    });

    const numWorkgroupsX = Math.ceil(M / workgroupSize);
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroupsX, batchSize);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
}

/**
 * Extract result and demodulate by chirp
 * output[k] = temp[k] * chirp[k] * scale for k < n
 */
function executeChirpDemodulatePass(
    device: GPUDevice,
    tempBuffer: GPUBuffer,
    chirpBuffer: GPUBuffer,
    outputBuffer: GPUBuffer,
    outputByteOffset: number,
    n: number,
    M: number,
    batchSize: number,
    scale: number,
    workgroupSize: number
): void {
    const shaderCode = buildChirpDemodulateShader({ workgroupSize });

    const pipelineKey = `bluestein_demod_wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    const uniformData = new ArrayBuffer(16);
    const u32View = new Uint32Array(uniformData);
    const f32View = new Float32Array(uniformData);
    u32View[0] = n;
    u32View[1] = M;
    u32View[2] = batchSize;
    f32View[3] = scale;

    const uniformBuffer = createUniformBuffer(uniformData);

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: tempBuffer } },
            { binding: 1, resource: { buffer: chirpBuffer } },
            { binding: 2, resource: { buffer: outputBuffer, offset: outputByteOffset } },
            { binding: 3, resource: { buffer: uniformBuffer } },
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

// === Shader builders ===

function buildChirpModulateShader(opts: { workgroupSize: number; isRealInput: boolean }): string {
    const { workgroupSize, isRealInput } = opts;
    const inputType = isRealInput ? 'f32' : 'vec2<f32>';

    return `
struct Params {
    n: u32,
    M: u32,
    batch_size: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<${inputType}>;
@group(0) @binding(1) var<storage, read> chirp: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let batch = gid.y;
    
    if (k >= params.M) { return; }
    
    let out_idx = batch * params.M + k;
    
    if (k < params.n) {
        let in_idx = batch * params.n + k;
        ${isRealInput ? `
        let x = vec2<f32>(input[in_idx], 0.0);
        ` : `
        let x = input[in_idx];
        `}
        let c = chirp[k];
        // Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        output[out_idx] = vec2<f32>(
            x.x * c.x - x.y * c.y,
            x.x * c.y + x.y * c.x
        );
    } else {
        output[out_idx] = vec2<f32>(0.0, 0.0);
    }
}
`;
}

function buildComplexMultiplyShader(opts: { workgroupSize: number }): string {
    return `
struct Params {
    M: u32,
    batch_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> inputA: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> inputB: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${opts.workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let batch = gid.y;
    
    if (k >= params.M) { return; }
    
    let batch_idx = batch * params.M + k;
    let a = inputA[batch_idx];
    let b = inputB[k];  // Kernel is shared across batches
    
    // Complex multiply
    output[batch_idx] = vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}
`;
}

function buildChirpDemodulateShader(opts: { workgroupSize: number }): string {
    return `
struct Params {
    n: u32,
    M: u32,
    batch_size: u32,
    scale: f32,
}

@group(0) @binding(0) var<storage, read> temp: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> chirp: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${opts.workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let batch = gid.y;
    
    if (k >= params.n) { return; }
    
    let temp_idx = batch * params.M + k;
    let out_idx = batch * params.n + k;
    
    let t = temp[temp_idx];
    let c = chirp[k];
    
    // Complex multiply and scale
    let result = vec2<f32>(
        t.x * c.x - t.y * c.y,
        t.x * c.y + t.y * c.x
    );
    
    output[out_idx] = result * params.scale;
}
`;
}

/**
 * Execute Bluestein RFFT for non-power-of-2 real input
 * 
 * Uses Bluestein FFT on real input and outputs onesided spectrum
 */
export function executeBluesteinRFFT(args: RFFTKernelArgs): void {
    const { input, output, dim, n, norm, onesidedLen } = args;

    const inputTensor = input as WebGPUTensor<DType>;
    const outputTensor = output as WebGPUTensor<DType>;

    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;

    // Calculate padded size M >= 2N-1, must be power of 2
    const M = nextPowerOf2(2 * n - 1);
    const log2M = Math.log2(M);

    // Calculate batch size
    const inputShape = input.shape;
    let batchSize = 1;
    for (let i = 0; i < inputShape.length; i++) {
        if (i !== dim) {
            batchSize *= inputShape[i];
        }
    }

    // Get GPU buffers and offsets
    const inputBuffer = inputTensor.storage.buffer as GPUBuffer;
    const outputBuffer = outputTensor.storage.buffer as GPUBuffer;

    const resolver = getGlobalDTypeResolver();
    const inputBytesPerElement = resolver.getDescriptor(input.dtype).gpuBytesPerElement;
    const outputBytesPerElement = resolver.getDescriptor(output.dtype).gpuBytesPerElement;

    const inputByteOffset = inputTensor.offset * inputBytesPerElement;
    const outputByteOffset = outputTensor.offset * outputBytesPerElement;

    // === Step 1: Generate chirp sequences on CPU ===
    const direction = 'forward';
    const chirp = generateChirp(n, direction);
    const chirpKernel = generateChirpKernel(n, M, direction);

    // Upload chirp to GPU
    const chirpBuffer = device.createBuffer({
        size: chirp.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(chirpBuffer, 0, chirp.buffer);

    // === Step 2: Create temporary buffers ===
    const bufferASize = M * batchSize * 8;
    const bufferA = device.createBuffer({
        size: bufferASize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const bufferBSize = M * 8;
    const bufferB = device.createBuffer({
        size: bufferBSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // === Step 3: Prepare chirp kernel FFT ===
    const chirpKernelBuffer = device.createBuffer({
        size: chirpKernel.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(chirpKernelBuffer, 0, chirpKernel.buffer);

    executeBitReversePass(
        device, chirpKernelBuffer, bufferB, 0, 0,
        M, log2M, 1,
        0, 0, 1, 1,
        false, workgroupSize
    );

    const twiddleBufferM = getTwiddleBuffer(device, M);
    for (let stage = 0; stage < log2M; stage++) {
        executeButterflyPass(device, bufferB, twiddleBufferM, 0, M, stage, -1.0, 1, workgroupSize);
    }

    chirpKernelBuffer.destroy();

    // === Step 4: Modulate real input and zero-pad ===
    executeChirpModulatePass(
        device,
        inputBuffer,
        chirpBuffer,
        bufferA,
        inputByteOffset,
        n,
        M,
        batchSize,
        true, // isRealInput
        workgroupSize
    );

    // === Step 5: FFT on modulated input ===
    const tempBuffer = device.createBuffer({
        size: bufferASize,
        usage: GPUBufferUsage.STORAGE,
    });

    executeBitReversePass(
        device, bufferA, tempBuffer, 0, 0,
        M, log2M, batchSize,
        M, 0, 1, 1,
        false, workgroupSize
    );

    for (let stage = 0; stage < log2M; stage++) {
        executeButterflyPass(device, tempBuffer, twiddleBufferM, 0, M, stage, -1.0, batchSize, workgroupSize);
    }

    // === Step 6: Element-wise multiply temp * bufferB ===
    executeComplexMultiplyPass(
        device,
        tempBuffer,
        bufferB,
        bufferA,
        M,
        batchSize,
        workgroupSize
    );

    // === Step 7: IFFT on result ===
    executeBitReversePass(
        device, bufferA, tempBuffer, 0, 0,
        M, log2M, batchSize,
        M, 0, 1, 1,
        false, workgroupSize
    );

    for (let stage = 0; stage < log2M; stage++) {
        executeButterflyPass(device, tempBuffer, twiddleBufferM, 0, M, stage, 1.0, batchSize, workgroupSize);
    }

    executeScalePass(device, tempBuffer, 0, M * batchSize, 1.0 / M, workgroupSize);

    // === Step 8: Extract onesided and demodulate ===
    // Create full spectrum buffer first
    const fullSpectrumSize = n * batchSize * 8;
    const fullSpectrumBuffer = device.createBuffer({
        size: fullSpectrumSize,
        usage: GPUBufferUsage.STORAGE,
    });

    const finalScale = computeNormalizationScale(n, norm, 'forward');

    executeChirpDemodulatePass(
        device,
        tempBuffer,
        chirpBuffer,
        fullSpectrumBuffer,
        0,
        n,
        M,
        batchSize,
        finalScale,
        workgroupSize
    );

    // === Step 9: Copy onesided (first onesidedLen elements) to output ===
    executeExtractOnesidedPass(
        device,
        fullSpectrumBuffer,
        outputBuffer,
        outputByteOffset,
        n,
        onesidedLen,
        batchSize,
        workgroupSize
    );

    // Cleanup
    chirpBuffer.destroy();
    bufferA.destroy();
    bufferB.destroy();
    tempBuffer.destroy();
    fullSpectrumBuffer.destroy();
}

/**
 * Execute Bluestein IRFFT for non-power-of-2 output
 * 
 * Reconstructs full spectrum from onesided, uses Bluestein IFFT, extracts real part
 */
export function executeBluesteinIRFFT(args: IRFFTKernelArgs): void {
    const { input, output, dim, n, norm, onesidedLen } = args;

    const inputTensor = input as WebGPUTensor<DType>;
    const outputTensor = output as WebGPUTensor<DType>;

    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;

    // Calculate padded size M >= 2N-1, must be power of 2
    const M = nextPowerOf2(2 * n - 1);
    const log2M = Math.log2(M);

    // Calculate batch size
    const inputShape = input.shape;
    let batchSize = 1;
    for (let i = 0; i < inputShape.length; i++) {
        if (i !== dim) {
            batchSize *= inputShape[i];
        }
    }

    // Get GPU buffers and offsets
    const inputBuffer = inputTensor.storage.buffer as GPUBuffer;
    const outputBuffer = outputTensor.storage.buffer as GPUBuffer;

    const resolver = getGlobalDTypeResolver();
    const inputBytesPerElement = resolver.getDescriptor(input.dtype).gpuBytesPerElement;
    const outputBytesPerElement = resolver.getDescriptor(output.dtype).gpuBytesPerElement;

    const inputByteOffset = inputTensor.offset * inputBytesPerElement;
    const outputByteOffset = outputTensor.offset * outputBytesPerElement;

    // === Step 1: Reconstruct full spectrum from onesided ===
    const fullSpectrumSize = n * batchSize * 8;
    const fullSpectrumBuffer = device.createBuffer({
        size: fullSpectrumSize,
        usage: GPUBufferUsage.STORAGE,
    });

    executeHermitianMirrorBluesteinPass(
        device,
        inputBuffer,
        fullSpectrumBuffer,
        inputByteOffset,
        n,
        onesidedLen,
        batchSize,
        workgroupSize
    );

    // === Step 2: Generate chirp sequences for inverse ===
    const direction = 'inverse';
    const chirp = generateChirp(n, direction);
    const chirpKernel = generateChirpKernel(n, M, direction);

    const chirpBuffer = device.createBuffer({
        size: chirp.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(chirpBuffer, 0, chirp.buffer);

    // === Step 3: Create temporary buffers ===
    const bufferASize = M * batchSize * 8;
    const bufferA = device.createBuffer({
        size: bufferASize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const bufferBSize = M * 8;
    const bufferB = device.createBuffer({
        size: bufferBSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // === Step 4: Prepare chirp kernel FFT ===
    const chirpKernelBuffer = device.createBuffer({
        size: chirpKernel.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(chirpKernelBuffer, 0, chirpKernel.buffer);

    executeBitReversePass(
        device, chirpKernelBuffer, bufferB, 0, 0,
        M, log2M, 1,
        0, 0, 1, 1,
        false, workgroupSize
    );

    const twiddleBufferM = getTwiddleBuffer(device, M);
    for (let stage = 0; stage < log2M; stage++) {
        executeButterflyPass(device, bufferB, twiddleBufferM, 0, M, stage, -1.0, 1, workgroupSize);
    }

    chirpKernelBuffer.destroy();

    // === Step 5: Modulate full spectrum and zero-pad ===
    executeChirpModulatePass(
        device,
        fullSpectrumBuffer,
        chirpBuffer,
        bufferA,
        0,
        n,
        M,
        batchSize,
        false, // isRealInput = false (complex input)
        workgroupSize
    );

    // === Step 6: FFT on modulated input ===
    const tempBuffer = device.createBuffer({
        size: bufferASize,
        usage: GPUBufferUsage.STORAGE,
    });

    executeBitReversePass(
        device, bufferA, tempBuffer, 0, 0,
        M, log2M, batchSize,
        M, 0, 1, 1,
        false, workgroupSize
    );

    for (let stage = 0; stage < log2M; stage++) {
        executeButterflyPass(device, tempBuffer, twiddleBufferM, 0, M, stage, -1.0, batchSize, workgroupSize);
    }

    // === Step 7: Element-wise multiply ===
    executeComplexMultiplyPass(
        device,
        tempBuffer,
        bufferB,
        bufferA,
        M,
        batchSize,
        workgroupSize
    );

    // === Step 8: IFFT on result ===
    executeBitReversePass(
        device, bufferA, tempBuffer, 0, 0,
        M, log2M, batchSize,
        M, 0, 1, 1,
        false, workgroupSize
    );

    for (let stage = 0; stage < log2M; stage++) {
        executeButterflyPass(device, tempBuffer, twiddleBufferM, 0, M, stage, 1.0, batchSize, workgroupSize);
    }

    executeScalePass(device, tempBuffer, 0, M * batchSize, 1.0 / M, workgroupSize);

    // === Step 9: Demodulate and extract real part ===
    const finalScale = computeNormalizationScale(n, norm, 'inverse');

    executeChirpDemodulateRealPass(
        device,
        tempBuffer,
        chirpBuffer,
        outputBuffer,
        outputByteOffset,
        n,
        M,
        batchSize,
        finalScale,
        workgroupSize
    );

    // Cleanup
    chirpBuffer.destroy();
    bufferA.destroy();
    bufferB.destroy();
    tempBuffer.destroy();
    fullSpectrumBuffer.destroy();
}

// === Additional shader passes for RFFT/IRFFT ===

function executeExtractOnesidedPass(
    device: GPUDevice,
    srcBuffer: GPUBuffer,
    dstBuffer: GPUBuffer,
    dstByteOffset: number,
    n: number,
    onesidedLen: number,
    batchSize: number,
    workgroupSize: number
): void {
    const shaderCode = `
struct Params {
    n: u32,
    onesided_len: u32,
    batch_size: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let batch = gid.y;
    
    if (k >= params.onesided_len) { return; }
    
    let src_idx = batch * params.n + k;
    let dst_idx = batch * params.onesided_len + k;
    
    dst[dst_idx] = src[src_idx];
}
`;

    const pipelineKey = `bluestein_extract_onesided_wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    const uniformData = new ArrayBuffer(16);
    const u32View = new Uint32Array(uniformData);
    u32View[0] = n;
    u32View[1] = onesidedLen;
    u32View[2] = batchSize;
    u32View[3] = 0;

    const uniformBuffer = createUniformBuffer(uniformData);

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: srcBuffer } },
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

function executeHermitianMirrorBluesteinPass(
    device: GPUDevice,
    srcBuffer: GPUBuffer,
    dstBuffer: GPUBuffer,
    srcByteOffset: number,
    n: number,
    onesidedLen: number,
    batchSize: number,
    workgroupSize: number
): void {
    const shaderCode = `
struct Params {
    n: u32,
    onesided_len: u32,
    batch_size: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let batch = gid.y;
    
    if (k >= params.n) { return; }
    
    let dst_idx = batch * params.n + k;
    
    if (k < params.onesided_len) {
        // Direct copy for first half
        let src_idx = batch * params.onesided_len + k;
        dst[dst_idx] = src[src_idx];
    } else {
        // Hermitian symmetry: X[k] = conj(X[N-k])
        let mirror_k = params.n - k;
        let src_idx = batch * params.onesided_len + mirror_k;
        let val = src[src_idx];
        dst[dst_idx] = vec2<f32>(val.x, -val.y);  // Conjugate
    }
}
`;

    const pipelineKey = `bluestein_hermitian_mirror_wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    const uniformData = new ArrayBuffer(16);
    const u32View = new Uint32Array(uniformData);
    u32View[0] = n;
    u32View[1] = onesidedLen;
    u32View[2] = batchSize;
    u32View[3] = 0;

    const uniformBuffer = createUniformBuffer(uniformData);

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: srcBuffer, offset: srcByteOffset } },
            { binding: 1, resource: { buffer: dstBuffer } },
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

function executeChirpDemodulateRealPass(
    device: GPUDevice,
    tempBuffer: GPUBuffer,
    chirpBuffer: GPUBuffer,
    outputBuffer: GPUBuffer,
    outputByteOffset: number,
    n: number,
    M: number,
    batchSize: number,
    scale: number,
    workgroupSize: number
): void {
    const shaderCode = `
struct Params {
    n: u32,
    M: u32,
    batch_size: u32,
    scale: f32,
}

@group(0) @binding(0) var<storage, read> temp: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> chirp: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let batch = gid.y;
    
    if (k >= params.n) { return; }
    
    let temp_idx = batch * params.M + k;
    let out_idx = batch * params.n + k;
    
    let t = temp[temp_idx];
    let c = chirp[k];
    
    // Complex multiply and take real part
    let result_real = (t.x * c.x - t.y * c.y) * params.scale;
    
    output[out_idx] = result_real;
}
`;

    const pipelineKey = `bluestein_demod_real_wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    const uniformData = new ArrayBuffer(16);
    const u32View = new Uint32Array(uniformData);
    const f32View = new Float32Array(uniformData);
    u32View[0] = n;
    u32View[1] = M;
    u32View[2] = batchSize;
    f32View[3] = scale;

    const uniformBuffer = createUniformBuffer(uniformData);

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: tempBuffer } },
            { binding: 1, resource: { buffer: chirpBuffer } },
            { binding: 2, resource: { buffer: outputBuffer, offset: outputByteOffset } },
            { binding: 3, resource: { buffer: uniformBuffer } },
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

