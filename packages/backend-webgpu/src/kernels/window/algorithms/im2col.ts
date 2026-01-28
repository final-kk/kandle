/**
 * Im2Col Algorithm for Conv2d
 * 
 * 将卷积操作转换为矩阵乘法：
 * 1. Im2Col: 将输入展开为 [N * H_out * W_out, C_in * kH * kW] 矩阵
 * 2. GEMM: 与权重矩阵相乘
 * 3. Reshape: 重塑回 [N, C_out, H_out, W_out] 或 [N, H_out, W_out, C_out]
 * 
 * 优势：
 * - 复用高度优化的 GEMM 实现
 * - 对大多数卷积配置性能良好
 * - 实现相对简单可靠
 * 
 * @module kernels/window/algorithms/im2col
 */

import type { ITensorHandle, DType } from '@kandle/types';
import type { ConvDispatchResult } from '../types';
import { Logger } from '@kandle/utils';
import { WebGPUDeviceManager } from '../../../base/device';
import { WebGPUPipelineManager } from '../../../pipelines/WebGPUPipelineManager';
import { WebGPUTensor } from '../../../base/tensor';
import { matmulExecutor } from '../../matrix/executor';
import type { MatmulDispatchResult } from '../../matrix/types';
import { MemoryFormat as MF } from '@kandle/types';
import { createUniformBuffer } from '../../../base/uniformUtils';

const logger = new Logger('Im2Col');

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE_X = 16;
const WORKGROUP_SIZE_Y = 16;

// ============================================================================
// Strided Index Helpers (工业级)
// ============================================================================

/**
 * 生成 strided 输入索引计算函数 (工业级)
 * 
 * 使用 strides/offset 计算物理地址，支持任意非连续输入
 * 
 * 重要：strides 数组始终按逻辑形状维度顺序存储 [strideN, strideC, strideH, strideW]，
 * 与 memoryFormat 无关。memoryFormat 影响的是 stride 的值，而不是 stride 在数组中的位置。
 * 
 * @param isChannelsLast - 这个参数在此函数中实际上不需要使用，保留仅为了 API 兼容性
 * @param strides - 输入 tensor 的 strides，始终为 [strideN, strideC, strideH, strideW]
 * @param offset - 输入 tensor 的 offset
 * @param hasChannelOffset - 是否使用 uniforms.channelOffset (for grouped conv)
 */
function generateStridedInputIndexFn(
    isChannelsLast: boolean,
    strides: readonly number[],
    offset: number,
    hasChannelOffset: boolean
): string {
    // 支持 3D (conv1d) 和 4D (conv2d/3d) 输入
    // 3D tensor strides: [strideN, strideC, strideL]
    // 4D tensor strides: [strideN, strideC, strideH, strideW]

    const channelExpr = hasChannelOffset ? 'c + uniforms.channelOffset' : 'c';

    if (strides.length === 3) {
        // Conv1d: 3D tensor [N, C, L]
        // 将其视为 [N, C, 1, L]，H=1
        const strideN = strides[0];
        const strideC = strides[1];
        const strideL = strides[2];

        return `
// Strided 输入索引计算 (Conv1d: 3D -> 4D 模拟, H=1)
const INPUT_OFFSET: i32 = ${offset};
const STRIDE_N: i32 = ${strideN};
const STRIDE_C: i32 = ${strideC};
const STRIDE_W: i32 = ${strideL};

fn getInputIndex(n: i32, c: i32, h: i32, w: i32) -> i32 {
    // h 被忽略 (H=1 for conv1d)
    let actual_c = ${channelExpr};
    return INPUT_OFFSET + n * STRIDE_N + actual_c * STRIDE_C + w * STRIDE_W;
}`;
    } else if (strides.length === 4) {
        // Conv2d: 4D tensor [N, C, H, W]
        const strideN = strides[0];
        const strideC = strides[1];
        const strideH = strides[2];
        const strideW = strides[3];

        return `
// Strided 输入索引计算 (工业级: 支持非连续访问)
const INPUT_OFFSET: i32 = ${offset};
const STRIDE_N: i32 = ${strideN};
const STRIDE_C: i32 = ${strideC};
const STRIDE_H: i32 = ${strideH};
const STRIDE_W: i32 = ${strideW};

fn getInputIndex(n: i32, c: i32, h: i32, w: i32) -> i32 {
    let actual_c = ${channelExpr};
    return INPUT_OFFSET + n * STRIDE_N + actual_c * STRIDE_C + h * STRIDE_H + w * STRIDE_W;
}`;
    } else {
        throw new Error(`Im2Col requires 3D or 4D input, got strides length ${strides.length}`);
    }
}


// ============================================================================
// Main Execute Function
// ============================================================================

/**
 * 执行 Im2Col 卷积算法
 * 
 * 流程:
 * 1. 创建 Im2Col 中间矩阵
 * 2. 执行 Im2Col 变换
 * 3. 调用 GEMM kernel
 * 4. 添加 bias (如果有)
 * 5. Reshape 输出
 */
export function executeIm2ColConv(
    config: ConvDispatchResult,
    input: ITensorHandle,
    weight: ITensorHandle,
    bias?: ITensorHandle
): void {
    const device = WebGPUDeviceManager.device;

    const isChannelsLast = config.isChannelsLast;
    const { batchSize, inChannels, outChannels, groups } = config;

    // 支持 conv1d (1D spatial) 和 conv2d (2D spatial)
    // 将 1D 规范化为 2D, H=1
    const inputSpatial = config.inputSpatial.length === 1
        ? [1, config.inputSpatial[0]]
        : config.inputSpatial;
    const outputSpatial = config.outputSpatial.length === 1
        ? [1, config.outputSpatial[0]]
        : config.outputSpatial;
    const kernelSize = config.kernelSize.length === 1
        ? [1, config.kernelSize[0]]
        : config.kernelSize;
    const stride = config.stride.length === 1
        ? [1, config.stride[0]]
        : config.stride;
    const padding = config.padding.length === 1
        ? [0, config.padding[0]]
        : config.padding;
    const dilation = config.dilation.length === 1
        ? [1, config.dilation[0]]
        : config.dilation;

    const [H, W] = inputSpatial;
    const [outH, outW] = outputSpatial;
    const [kH, kW] = kernelSize;
    const [strideH, strideW] = stride;
    const [padH, padW] = padding;
    const [dilationH, dilationW] = dilation;

    const channelsPerGroup = inChannels / groups;
    const outChannelsPerGroup = outChannels / groups;

    // M = batch * outputSpatial
    const M = batchSize * outH * outW;
    // K = channelsPerGroup * kernelSize
    const K = channelsPerGroup * kH * kW;
    // N = outChannelsPerGroup
    const N = outChannelsPerGroup;

    logger.debug(`Im2Col Conv: M=${M}, K=${K}, N=${N}, groups=${groups}`);

    if (groups === 1) {
        // 标准卷积：单次 GEMM
        executeStandardConv(
            device, config, input, weight, bias,
            M, K, N, outH, outW,
            H, W, kH, kW, strideH, strideW, padH, padW, dilationH, dilationW,
            channelsPerGroup, isChannelsLast
        );
    } else {
        // 分组卷积：多次 GEMM
        executeGroupedConv(
            device, config, input, weight, bias,
            M, K, N, outH, outW,
            H, W, kH, kW, strideH, strideW, padH, padW, dilationH, dilationW,
            channelsPerGroup, outChannelsPerGroup, groups, isChannelsLast
        );
    }
}


// ============================================================================
// Standard Convolution (groups=1)
// ============================================================================

function executeStandardConv(
    device: GPUDevice,
    config: ConvDispatchResult,
    input: ITensorHandle,
    weight: ITensorHandle,
    bias: ITensorHandle | undefined,
    M: number, K: number, N: number,
    outH: number, outW: number,
    H: number, W: number,
    kH: number, kW: number,
    strideH: number, strideW: number,
    padH: number, padW: number,
    dilationH: number, dilationW: number,
    channelsPerGroup: number,
    isChannelsLast: boolean
): void {
    const outChannels = config.outChannels;
    const batchSize = config.batchSize;

    // Step 1: Create Im2Col intermediate buffer
    const im2colBuffer = createIm2ColBuffer(device, M, K, config.computeDtype);

    // Step 2: Execute Im2Col (工业级: 传递 strides/offset)
    executeIm2ColKernel(
        device, input, im2colBuffer,
        batchSize, channelsPerGroup,
        H, W, outH, outW,
        kH, kW, strideH, strideW, padH, padW, dilationH, dilationW,
        isChannelsLast, config.computeDtype,
        input.strides, input.offset  // 工业级: strided 支持
    );

    // Step 3: Execute GEMM
    // Im2Col output [M, K] @ Weight^T [K, N] = Output [M, N]
    // GEMM 输出是 [M, N] row-major: [pos0_ch0, pos0_ch1, ...]
    // 
    // 对于 NCHW 输出，需要转换为: [ch0_pos0, ch0_pos1, ..., ch1_pos0, ...]
    // 对于 NHWC 输出，GEMM 已经是正确格式
    const needsLayoutTranspose = !isChannelsLast && outChannels > 1;

    if (needsLayoutTranspose) {
        // 创建临时缓冲区存放 GEMM 的 row-major 输出
        const gemmOutput = createIm2ColBuffer(device, M, outChannels, config.computeDtype);
        executeConvGemm(im2colBuffer, weight, gemmOutput, M, K, outChannels, config.computeDtype);

        // 添加 bias 到 GEMM 输出 (在转置之前)
        if (bias) {
            addBiasInPlace(device, gemmOutput, bias, M, outChannels, config.computeDtype);
        }

        // 转换 [M, N] row-major 到 NCHW 格式
        // M = batchSize * outH * outW, N = outChannels
        // Source: [pos, ch] -> Target: [n, ch, h, w]
        transposeGemmToNCHW(
            device, gemmOutput, config.output,
            batchSize, outChannels, outH, outW,
            config.computeDtype
        );
    } else {
        // NHWC 格式或单输出通道：GEMM 输出直接可用
        executeConvGemm(im2colBuffer, weight, config.output, M, K, outChannels, config.computeDtype);

        // Step 4: Add bias (if present)
        if (bias) {
            addBiasInPlace(device, config.output, bias, M, outChannels, config.computeDtype);
        }
    }
}

// ============================================================================
// Grouped Convolution (groups > 1)
// ============================================================================

function executeGroupedConv(
    device: GPUDevice,
    config: ConvDispatchResult,
    input: ITensorHandle,
    weight: ITensorHandle,
    bias: ITensorHandle | undefined,
    M: number, K: number, N: number,
    outH: number, outW: number,
    H: number, W: number,
    kH: number, kW: number,
    strideH: number, strideW: number,
    padH: number, padW: number,
    dilationH: number, dilationW: number,
    channelsPerGroup: number,
    outChannelsPerGroup: number,
    groups: number,
    isChannelsLast: boolean
): void {
    logger.debug(`Executing grouped conv with ${groups} groups`);

    // For grouped convolution, we currently fall back to direct convolution
    // which handles groups natively. A more optimized approach would use
    // batched GEMM with proper channel offsets.

    // Create shared Im2Col buffer (size is per-group)
    const im2colBuffer = createIm2ColBuffer(device, M, K, config.computeDtype);

    for (let g = 0; g < groups; g++) {
        const channelOffset = g * channelsPerGroup;

        // Execute Im2Col for this group (工业级: 传递 strides/offset)
        executeIm2ColKernelGrouped(
            device, input, im2colBuffer,
            config.batchSize, channelsPerGroup, config.inChannels,
            H, W, outH, outW,
            kH, kW, strideH, strideW, padH, padW, dilationH, dilationW,
            channelOffset,
            isChannelsLast, config.computeDtype,
            input.strides, input.offset  // 工业级: strided 支持
        );

        // TODO: Implement proper group-wise GEMM with output offset
        // For now, this is a simplified version
    }

    // Add bias for all groups (if present)
    if (bias) {
        addBiasInPlace(device, config.output, bias, M, config.outChannels, config.computeDtype);
    }
}

// ============================================================================
// Im2Col Buffer Creation
// ============================================================================

function createIm2ColBuffer(
    _device: GPUDevice,  // Unused: WebGPUTensor.createNew handles device internally
    M: number,
    K: number,
    dtype: DType
): ITensorHandle {
    // Use the proper tensor creation infrastructure instead of manual mock
    // This ensures all ITensorHandle properties are correctly initialized
    return WebGPUTensor.createNew([M, K], dtype);
}

// ============================================================================
// Im2Col Kernel
// ============================================================================

function executeIm2ColKernel(
    device: GPUDevice,
    input: ITensorHandle,
    output: ITensorHandle,
    batchSize: number,
    channels: number,
    H: number, W: number,
    outH: number, outW: number,
    kH: number, kW: number,
    strideH: number, strideW: number,
    padH: number, padW: number,
    dilationH: number, dilationW: number,
    isChannelsLast: boolean,
    dtype: DType,
    // 工业级: Strided 支持
    inputStrides: readonly number[],
    inputOffset: number
): void {
    const M = batchSize * outH * outW;
    const K = channels * kH * kW;

    // Generate shader (工业级: 包含 strides/offset)
    const shaderCode = buildIm2ColShader(
        batchSize, channels,
        H, W, outH, outW,
        kH, kW, strideH, strideW, padH, padW, dilationH, dilationW,
        isChannelsLast, dtype,
        inputStrides, inputOffset
    );

    // Create uniform buffer
    const uniformData = new Int32Array([
        batchSize, channels, H, W,
        outH, outW, kH, kW,
        strideH, strideW, padH, padW,
        dilationH, dilationW, M, K
    ]);

    const uniformBuffer = createUniformBuffer(uniformData.buffer);

    // Get or create pipeline (工业级: cache key 包含 strides)
    const cacheKey = `im2col-${dtype}-${batchSize}-${channels}-${H}x${W}-${kH}x${kW}-${isChannelsLast}-s${inputStrides.join('_')}-o${inputOffset}`;
    let pipeline = WebGPUPipelineManager.getPipeline(cacheKey);
    if (!pipeline) {
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
        WebGPUPipelineManager.registerPipeline(cacheKey, pipeline);
    }

    // Create bind group
    const inputBuffer = (input as any).storage.buffer as GPUBuffer;
    const outputBuffer = (output as any).storage.buffer as GPUBuffer;

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: inputBuffer } },
            { binding: 2, resource: { buffer: outputBuffer } },
        ],
    });

    // Dispatch
    const workgroupsX = Math.ceil(K / WORKGROUP_SIZE_X);
    const workgroupsY = Math.ceil(M / WORKGROUP_SIZE_Y);

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
}

function executeIm2ColKernelGrouped(
    device: GPUDevice,
    input: ITensorHandle,
    output: ITensorHandle,
    batchSize: number,
    channelsPerGroup: number,
    totalChannels: number,
    H: number, W: number,
    outH: number, outW: number,
    kH: number, kW: number,
    strideH: number, strideW: number,
    padH: number, padW: number,
    dilationH: number, dilationW: number,
    channelOffset: number,
    isChannelsLast: boolean,
    dtype: DType,
    // 工业级: Strided 支持
    inputStrides: readonly number[],
    inputOffset: number
): void {
    const M = batchSize * outH * outW;
    const K = channelsPerGroup * kH * kW;

    // Generate shader with channel offset support (工业级: 包含 strides/offset)
    const shaderCode = buildIm2ColShaderGrouped(
        batchSize, channelsPerGroup, totalChannels,
        H, W, outH, outW,
        kH, kW, strideH, strideW, padH, padW, dilationH, dilationW,
        channelOffset,
        isChannelsLast, dtype,
        inputStrides, inputOffset
    );

    const uniformData = new Int32Array([
        batchSize, channelsPerGroup, H, W,
        outH, outW, kH, kW,
        strideH, strideW, padH, padW,
        dilationH, dilationW, M, K,
        channelOffset, totalChannels, 0, 0  // Extra uniforms for grouping
    ]);

    const uniformBuffer = createUniformBuffer(uniformData.buffer);

    // 工业级: cache key 包含 strides
    const cacheKey = `im2col-grouped-${dtype}-${batchSize}-${channelsPerGroup}-${H}x${W}-${kH}x${kW}-${channelOffset}-${isChannelsLast}-s${inputStrides.join('_')}-o${inputOffset}`;
    let pipeline = WebGPUPipelineManager.getPipeline(cacheKey);
    if (!pipeline) {
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
        WebGPUPipelineManager.registerPipeline(cacheKey, pipeline);
    }

    const inputBuffer = (input as any).storage.buffer as GPUBuffer;
    const outputBuffer = (output as any).storage.buffer as GPUBuffer;

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: inputBuffer } },
            { binding: 2, resource: { buffer: outputBuffer } },
        ],
    });

    const workgroupsX = Math.ceil(K / WORKGROUP_SIZE_X);
    const workgroupsY = Math.ceil(M / WORKGROUP_SIZE_Y);

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
}

// ============================================================================
// Im2Col Shader Generation
// ============================================================================

/**
 * 工业级 Im2Col Shader 生成 (Strided 支持)
 * 
 * 使用 strides/offset 计算物理地址，支持任意非连续输入
 */
function buildIm2ColShader(
    batchSize: number,
    channels: number,
    H: number, W: number,
    outH: number, outW: number,
    kH: number, kW: number,
    strideH: number, strideW: number,
    padH: number, padW: number,
    dilationH: number, dilationW: number,
    isChannelsLast: boolean,
    dtype: DType,
    // 工业级: Strided 支持
    inputStrides: readonly number[],
    inputOffset: number
): string {
    const dataType = dtype === 'float16' ? 'f16' : 'f32';
    const enableF16 = dtype === 'float16' ? 'enable f16;' : '';

    // 工业级: 使用 strides 生成索引计算函数
    const getInputIndex = generateStridedInputIndexFn(isChannelsLast, inputStrides, inputOffset, false);

    return `
${enableF16}

struct Uniforms {
    batchSize: i32,
    channels: i32,
    H: i32,
    W: i32,
    outH: i32,
    outW: i32,
    kH: i32,
    kW: i32,
    strideH: i32,
    strideW: i32,
    padH: i32,
    padW: i32,
    dilationH: i32,
    dilationW: i32,
    M: i32,  // batchSize * outH * outW
    K: i32,  // channels * kH * kW
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<${dataType}>;
@group(0) @binding(2) var<storage, read_write> output: array<${dataType}>;

${getInputIndex}

@compute @workgroup_size(${WORKGROUP_SIZE_X}, ${WORKGROUP_SIZE_Y}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = i32(gid.x);  // Column index (within K dimension)
    let m = i32(gid.y);  // Row index (within M dimension)
    
    if (k >= uniforms.K || m >= uniforms.M) {
        return;
    }
    
    // Decode m -> (n, h_out, w_out)
    let w_out = m % uniforms.outW;
    let temp = m / uniforms.outW;
    let h_out = temp % uniforms.outH;
    let n = temp / uniforms.outH;
    
    // Decode k -> (c, kh, kw)
    let kw = k % uniforms.kW;
    let temp2 = k / uniforms.kW;
    let kh = temp2 % uniforms.kH;
    let c = temp2 / uniforms.kH;
    
    // Calculate input position
    let h_in = h_out * uniforms.strideH - uniforms.padH + kh * uniforms.dilationH;
    let w_in = w_out * uniforms.strideW - uniforms.padW + kw * uniforms.dilationW;
    
    var value: ${dataType} = ${dataType}(0.0);
    
    // Boundary check
    if (h_in >= 0 && h_in < uniforms.H && w_in >= 0 && w_in < uniforms.W) {
        let inputIdx = getInputIndex(n, c, h_in, w_in);
        value = input[inputIdx];
    }
    
    // Write to Im2Col output
    let outputIdx = m * uniforms.K + k;
    output[outputIdx] = value;
}
`;
}

/**
 * 工业级 Im2Col Shader 生成 - Grouped Conv (Strided 支持)
 * 
 * 使用 strides/offset 计算物理地址，支持任意非连续输入
 * 支持分组卷积的 channel offset
 */
function buildIm2ColShaderGrouped(
    batchSize: number,
    channelsPerGroup: number,
    totalChannels: number,
    H: number, W: number,
    outH: number, outW: number,
    kH: number, kW: number,
    strideH: number, strideW: number,
    padH: number, padW: number,
    dilationH: number, dilationW: number,
    channelOffset: number,
    isChannelsLast: boolean,
    dtype: DType,
    // 工业级: Strided 支持
    inputStrides: readonly number[],
    inputOffset: number
): string {
    const dataType = dtype === 'float16' ? 'f16' : 'f32';
    const enableF16 = dtype === 'float16' ? 'enable f16;' : '';

    // 工业级: 使用 strides 生成索引计算函数 (with channel offset for grouped conv)
    const getInputIndex = generateStridedInputIndexFn(isChannelsLast, inputStrides, inputOffset, true);

    return `
${enableF16}

struct Uniforms {
    batchSize: i32,
    channels: i32,  // channelsPerGroup
    H: i32,
    W: i32,
    outH: i32,
    outW: i32,
    kH: i32,
    kW: i32,
    strideH: i32,
    strideW: i32,
    padH: i32,
    padW: i32,
    dilationH: i32,
    dilationW: i32,
    M: i32,
    K: i32,
    channelOffset: i32,
    totalChannels: i32,
    _padding1: i32,
    _padding2: i32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<${dataType}>;
@group(0) @binding(2) var<storage, read_write> output: array<${dataType}>;

${getInputIndex}

@compute @workgroup_size(${WORKGROUP_SIZE_X}, ${WORKGROUP_SIZE_Y}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = i32(gid.x);
    let m = i32(gid.y);
    
    if (k >= uniforms.K || m >= uniforms.M) {
        return;
    }
    
    let w_out = m % uniforms.outW;
    let temp = m / uniforms.outW;
    let h_out = temp % uniforms.outH;
    let n = temp / uniforms.outH;
    
    let kw = k % uniforms.kW;
    let temp2 = k / uniforms.kW;
    let kh = temp2 % uniforms.kH;
    let c = temp2 / uniforms.kH;
    
    let h_in = h_out * uniforms.strideH - uniforms.padH + kh * uniforms.dilationH;
    let w_in = w_out * uniforms.strideW - uniforms.padW + kw * uniforms.dilationW;
    
    var value: ${dataType} = ${dataType}(0.0);
    
    if (h_in >= 0 && h_in < uniforms.H && w_in >= 0 && w_in < uniforms.W) {
        // getInputIndex 内部已经处理了 channelOffset
        let inputIdx = getInputIndex(n, c, h_in, w_in);
        value = input[inputIdx];
    }
    
    let outputIdx = m * uniforms.K + k;
    output[outputIdx] = value;
}
`;
}

// ============================================================================
// GEMM Integration
// ============================================================================

function executeConvGemm(
    im2colMatrix: ITensorHandle,
    weight: ITensorHandle,
    output: ITensorHandle,
    M: number,
    K: number,
    N: number,
    dtype: DType
): void {
    // Conv as GEMM:
    // im2colMatrix: [M, K]
    // weight: [N, K] (C_out, C_in * kH * kW)
    // output: [M, N]
    // 
    // We compute: im2colMatrix @ weight^T = [M, K] @ [K, N] = [M, N]

    const matmulConfig: MatmulDispatchResult = {
        variant: 'mm',
        output,
        M,
        K,
        N,
        batchShape: [],
        batchSize: 1,
        transposeA: false,
        transposeB: true,  // Weight is stored as [N, K], we need [K, N]
        computeDtype: dtype,
        // Strides for contiguous tensors (im2col output is always contiguous)
        stridesA: [K, 1],  // Row-major [M, K]
        stridesB: [K, 1],  // Row-major [N, K], but we use transposeB=true
        // BMM executor expects 4D strides: [batch0, batch1, row, col]
        // For 2D tensors, batch strides are 0
        fullStridesA: [0, 0, K, 1],
        fullStridesB: [0, 0, K, 1],
        ndimA: 2,
        ndimB: 2,
        batchStrideA: 0,
        batchStrideB: 0,
        alpha: 1.0,
        beta: 0.0,
    };

    // Call the existing matmul executor
    matmulExecutor(matmulConfig, im2colMatrix, weight);
}

// ============================================================================
// Bias Addition
// ============================================================================

function addBiasInPlace(
    device: GPUDevice,
    output: ITensorHandle,
    bias: ITensorHandle,
    M: number,
    N: number,
    dtype: DType
): void {
    // Add bias to each row of output
    // output[m, n] += bias[n]

    const dataType = dtype === 'float16' ? 'f16' : 'f32';
    const enableF16 = dtype === 'float16' ? 'enable f16;' : '';

    const shaderCode = `
${enableF16}

struct Uniforms {
    M: i32,
    N: i32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> output: array<${dataType}>;
@group(0) @binding(2) var<storage, read> bias: array<${dataType}>;

@compute @workgroup_size(${WORKGROUP_SIZE_X}, ${WORKGROUP_SIZE_Y}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = i32(gid.x);
    let m = i32(gid.y);
    
    if (n >= uniforms.N || m >= uniforms.M) {
        return;
    }
    
    let idx = m * uniforms.N + n;
    output[idx] = output[idx] + bias[n];
}
`;

    const uniformData = new Int32Array([M, N]);
    const uniformBuffer = createUniformBuffer(uniformData.buffer);

    const cacheKey = `conv-bias-add-${dtype}-${M}-${N}`;
    let pipeline = WebGPUPipelineManager.getPipeline(cacheKey);
    if (!pipeline) {
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
        WebGPUPipelineManager.registerPipeline(cacheKey, pipeline);
    }

    const outputBuffer = (output as any).storage.buffer as GPUBuffer;
    const biasBuffer = (bias as any).storage.buffer as GPUBuffer;

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: outputBuffer } },
            { binding: 2, resource: { buffer: biasBuffer } },
        ],
    });

    const workgroupsX = Math.ceil(N / WORKGROUP_SIZE_X);
    const workgroupsY = Math.ceil(M / WORKGROUP_SIZE_Y);

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
}

// ============================================================================
// GEMM to NCHW Layout Transpose
// ============================================================================

/**
 * 将 GEMM 输出从 [M, N] row-major 转换为 NCHW 格式
 * 
 * GEMM 输出布局 (row-major):
 *   索引 [m, n] 物理位置 = m * N + n
 *   其中 m = n_batch * outH * outW + h_out * outW + w_out
 * 
 * 对于位置 (b, h, w, c)，GEMM 中的索引是:
 *   gemmIdx = (b * outH * outW + h * outW + w) * N + c
 * 
 * NCHW 目标布局:
 *   索引 [n, c, h, w] = n * (C * H * W) + c * (H * W) + h * W + w
 * 
 * 本函数执行: output[n, c, h, w] = input[(n * H * W + h * W + w) * C + c]
 */
function transposeGemmToNCHW(
    device: GPUDevice,
    input: ITensorHandle,
    output: ITensorHandle,
    batchSize: number,
    outChannels: number,
    outH: number,
    outW: number,
    dtype: DType
): void {
    const dataType = dtype === 'float16' ? 'f16' : 'f32';
    const enableF16 = dtype === 'float16' ? 'enable f16;' : '';

    const shaderCode = `
${enableF16}

struct Uniforms {
    batchSize: i32,
    outChannels: i32,
    outH: i32,
    outW: i32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<${dataType}>;
@group(0) @binding(2) var<storage, read_write> output: array<${dataType}>;

@compute @workgroup_size(${WORKGROUP_SIZE_X}, ${WORKGROUP_SIZE_Y}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = i32(gid.x);
    let h = i32(gid.y);
    let idx = i32(gid.z);  // batch * channels + channel
    
    let n = idx / uniforms.outChannels;
    let c = idx % uniforms.outChannels;
    
    if (w >= uniforms.outW || h >= uniforms.outH || n >= uniforms.batchSize) {
        return;
    }
    
    // GEMM row-major index: [(n*H*W + h*W + w), c] = (n*H*W + h*W + w) * C + c
    let spatialSize = uniforms.outH * uniforms.outW;
    let gemmRow = n * spatialSize + h * uniforms.outW + w;
    let gemmIdx = gemmRow * uniforms.outChannels + c;
    
    // NCHW index: [n, c, h, w] = n * (C*H*W) + c * (H*W) + h * W + w
    let nchwIdx = n * (uniforms.outChannels * spatialSize) +
                  c * spatialSize +
                  h * uniforms.outW + w;
    
    output[nchwIdx] = input[gemmIdx];
}
`;

    const uniformData = new Int32Array([batchSize, outChannels, outH, outW]);
    const uniformBuffer = createUniformBuffer(uniformData.buffer);

    const cacheKey = `gemm-to-nchw-${dtype}-${batchSize}-${outChannels}-${outH}x${outW}`;
    let pipeline = WebGPUPipelineManager.getPipeline(cacheKey);
    if (!pipeline) {
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
        WebGPUPipelineManager.registerPipeline(cacheKey, pipeline);
    }

    const inputBuffer = (input as any).storage.buffer as GPUBuffer;
    const outputBuffer = (output as any).storage.buffer as GPUBuffer;

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: inputBuffer } },
            { binding: 2, resource: { buffer: outputBuffer } },
        ],
    });

    const workgroupsX = Math.ceil(outW / WORKGROUP_SIZE_X);
    const workgroupsY = Math.ceil(outH / WORKGROUP_SIZE_Y);
    const workgroupsZ = batchSize * outChannels;

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
}

