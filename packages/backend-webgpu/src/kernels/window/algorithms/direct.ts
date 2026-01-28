/**
 * Direct Convolution Algorithm
 * 
 * 直接计算卷积，不使用 Im2Col 变换。
 * 适用于：
 * - 1x1 卷积（等价于逐点乘法）
 * - 小 kernel 的分组/深度可分离卷积
 * 
 * 优势：
 * - 无额外内存开销（不需要 Im2Col 中间矩阵）
 * - 对 1x1 卷积和 depthwise 卷积非常高效
 * 
 * @module kernels/window/algorithms/direct
 */

import type { ITensorHandle, DType } from '@kandle/types';
import type { ConvDispatchResult } from '../types';
import { Logger } from '@kandle/utils';
import { WebGPUDeviceManager } from '../../../base/device';
import { WebGPUPipelineManager } from '../../../pipelines/WebGPUPipelineManager';
import { createUniformBuffer } from '../../../base/uniformUtils';

const logger = new Logger('DirectConv');

// ============================================================================
// Constants
// ============================================================================

const WORKGROUP_SIZE_X = 8;
const WORKGROUP_SIZE_Y = 8;

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
 */
function generateStridedInputIndexFn(
    isChannelsLast: boolean,
    strides: readonly number[],
    offset: number
): string {
    // 4D tensor strides 始终按逻辑形状顺序存储: [strideN, strideC, strideH, strideW]
    // memoryFormat 影响的是 stride 的值:
    // - NCHW: strides 可能是 [C*H*W, H*W, W, 1] (C 维度跨度大)
    // - NHWC: strides 可能是 [H*W*C, 1, W*C, C] (C 维度跨度小，最密集)
    // 但无论哪种格式，strides[0] 总是 N 维度的 stride，strides[1] 总是 C 维度的 stride，以此类推

    if (strides.length !== 4) {
        throw new Error(`Direct Conv requires 4D input, got strides length ${strides.length}`);
    }

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
    return INPUT_OFFSET + n * STRIDE_N + c * STRIDE_C + h * STRIDE_H + w * STRIDE_W;
}`;
}

// ============================================================================
// Main Execute Function
// ============================================================================

/**
 * 执行直接卷积
 * 
 * 对于 1x1 卷积，这等价于在空间位置上的矩阵乘法。
 * 对于更大的 kernel，使用标准的嵌套循环实现。
 */
export function executeDirectConv(
    config: ConvDispatchResult,
    input: ITensorHandle,
    weight: ITensorHandle,
    bias?: ITensorHandle
): void {
    const device = WebGPUDeviceManager.device;
    const [kH, kW] = config.kernelSize;

    if (kH === 1 && kW === 1) {
        // 1x1 convolution - optimized path
        execute1x1Conv(device, config, input, weight, bias);
    } else {
        // General direct convolution
        executeGeneralDirectConv(device, config, input, weight, bias);
    }
}

// ============================================================================
// 1x1 Convolution (Pointwise)
// ============================================================================

/**
 * 1x1 卷积 = 逐点变换
 * 
 * 每个空间位置独立执行 channel 到 channel 的线性变换
 */
function execute1x1Conv(
    device: GPUDevice,
    config: ConvDispatchResult,
    input: ITensorHandle,
    weight: ITensorHandle,
    bias?: ITensorHandle
): void {
    const { batchSize, inChannels, outChannels, groups } = config;
    const [H, W] = config.inputSpatial;
    const [outH, outW] = config.outputSpatial;
    const [strideH, strideW] = config.stride;
    const isChannelsLast = config.isChannelsLast;
    const dtype = config.computeDtype;

    const channelsPerGroup = inChannels / groups;
    const outChannelsPerGroup = outChannels / groups;

    logger.debug(`1x1 Conv: N=${batchSize}, C_in=${inChannels}, C_out=${outChannels}, HxW=${H}x${W}, groups=${groups}`);

    // 工业级: 获取输入 strides/offset
    const inputStrides = input.strides;
    const inputOffset = input.offset;

    // Generate shader (工业级: 包含 strides)
    const shaderCode = build1x1ConvShader(
        batchSize, inChannels, outChannels,
        H, W, outH, outW,
        strideH, strideW,
        channelsPerGroup, outChannelsPerGroup,
        isChannelsLast, dtype,
        bias !== undefined,
        inputStrides, inputOffset
    );

    // Create uniform buffer
    const uniformData = new Int32Array([
        batchSize, inChannels, outChannels,
        H, W, outH, outW,
        strideH, strideW,
        groups, channelsPerGroup, outChannelsPerGroup
    ]);

    const uniformBuffer = createUniformBuffer(uniformData.buffer);

    // Get or create pipeline (工业级: cache key 包含 strides)
    const cacheKey = `conv1x1-${dtype}-${batchSize}-${inChannels}-${outChannels}-${H}x${W}-g${groups}-${isChannelsLast}-${bias ? 'bias' : 'nobias'}-s${inputStrides.join('_')}-o${inputOffset}`;
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
    const weightBuffer = (weight as any).storage.buffer as GPUBuffer;
    const outputBuffer = (config.output as any).storage.buffer as GPUBuffer;

    const entries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: inputBuffer } },
        { binding: 2, resource: { buffer: weightBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
    ];

    if (bias) {
        const biasBuffer = (bias as any).storage.buffer as GPUBuffer;
        entries.push({ binding: 4, resource: { buffer: biasBuffer } });
    }

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries,
    });

    // Dispatch
    const totalOutputElements = batchSize * outChannels * outH * outW;
    const workgroups = Math.ceil(totalOutputElements / (WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y));

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(workgroups, 1, 1);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
}

/**
 * 工业级 1x1 Conv Shader 生成 (Strided 支持)
 */
function build1x1ConvShader(
    batchSize: number,
    inChannels: number,
    outChannels: number,
    H: number, W: number,
    outH: number, outW: number,
    strideH: number, strideW: number,
    channelsPerGroup: number,
    outChannelsPerGroup: number,
    isChannelsLast: boolean,
    dtype: DType,
    hasBias: boolean,
    // 工业级: Strided 支持
    inputStrides: readonly number[],
    inputOffset: number
): string {
    const dataType = dtype === 'float16' ? 'f16' : 'f32';
    const enableF16 = dtype === 'float16' ? 'enable f16;' : '';
    const totalOutputElements = batchSize * outChannels * outH * outW;

    // 工业级: 使用 strided 索引函数
    const inputIndexFn = generateStridedInputIndexFn(isChannelsLast, inputStrides, inputOffset);

    const outputIndexFn = isChannelsLast
        ? `fn getOutputIndex(n: i32, c: i32, h: i32, w: i32) -> i32 {
               return n * (uniforms.outH * uniforms.outW * uniforms.outChannels) +
                      h * (uniforms.outW * uniforms.outChannels) +
                      w * uniforms.outChannels + c;
           }`
        : `fn getOutputIndex(n: i32, c: i32, h: i32, w: i32) -> i32 {
               return n * (uniforms.outChannels * uniforms.outH * uniforms.outW) +
                      c * (uniforms.outH * uniforms.outW) +
                      h * uniforms.outW + w;
           }`;

    const weightIndexFn = `fn getWeightIndex(c_out: i32, c_in_offset: i32) -> i32 {
        return c_out * uniforms.channelsPerGroup + c_in_offset;
    }`;

    const biasBinding = hasBias
        ? `@group(0) @binding(4) var<storage, read> bias: array<${dataType}>;`
        : '';

    const biasAdd = hasBias
        ? 'sum = sum + bias[c_out];'
        : '';

    return `
${enableF16}

${inputIndexFn}

struct Uniforms {
    batchSize: i32,
    inChannels: i32,
    outChannels: i32,
    H: i32,
    W: i32,
    outH: i32,
    outW: i32,
    strideH: i32,
    strideW: i32,
    groups: i32,
    channelsPerGroup: i32,
    outChannelsPerGroup: i32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<${dataType}>;
@group(0) @binding(2) var<storage, read> weight: array<${dataType}>;
@group(0) @binding(3) var<storage, read_write> output: array<${dataType}>;
${biasBinding}

${outputIndexFn}
${weightIndexFn}

@compute @workgroup_size(${WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = i32(gid.x);
    
    if (idx >= ${totalOutputElements}) {
        return;
    }
    
    // Decode linear index to (n, c_out, h_out, w_out)
    let w_out = idx % uniforms.outW;
    var temp = idx / uniforms.outW;
    let h_out = temp % uniforms.outH;
    temp = temp / uniforms.outH;
    let c_out = temp % uniforms.outChannels;
    let n = temp / uniforms.outChannels;
    
    // Calculate input position (1x1 conv with stride)
    let h_in = h_out * uniforms.strideH;
    let w_in = w_out * uniforms.strideW;
    
    // Grouped convolution: determine which group
    let group = c_out / uniforms.outChannelsPerGroup;
    let c_in_start = group * uniforms.channelsPerGroup;
    
    var sum: ${dataType} = ${dataType}(0.0);
    
    for (var c_in_offset: i32 = 0; c_in_offset < uniforms.channelsPerGroup; c_in_offset++) {
        let c_in = c_in_start + c_in_offset;
        let inputVal = input[getInputIndex(n, c_in, h_in, w_in)];
        let weightVal = weight[getWeightIndex(c_out, c_in_offset)];
        sum = sum + inputVal * weightVal;
    }
    
    ${biasAdd}
    
    output[getOutputIndex(n, c_out, h_out, w_out)] = sum;
}
`;
}

// ============================================================================
// General Direct Convolution
// ============================================================================

/**
 * 通用直接卷积
 */
function executeGeneralDirectConv(
    device: GPUDevice,
    config: ConvDispatchResult,
    input: ITensorHandle,
    weight: ITensorHandle,
    bias?: ITensorHandle
): void {
    const { batchSize, inChannels, outChannels, groups } = config;
    const [H, W] = config.inputSpatial;
    const [outH, outW] = config.outputSpatial;
    const [kH, kW] = config.kernelSize;
    const [strideH, strideW] = config.stride;
    const [padH, padW] = config.padding;
    const [dilationH, dilationW] = config.dilation;
    const isChannelsLast = config.isChannelsLast;
    const dtype = config.computeDtype;

    const channelsPerGroup = inChannels / groups;
    const outChannelsPerGroup = outChannels / groups;

    logger.debug(`Direct Conv: N=${batchSize}, C_in=${inChannels}, C_out=${outChannels}, kernel=${kH}x${kW}, groups=${groups}`);

    // 工业级: 获取输入 strides/offset
    const inputStrides = input.strides;
    const inputOffset = input.offset;

    // Generate shader (工业级: 包含 strides)
    const shaderCode = buildDirectConvShader(
        batchSize, inChannels, outChannels,
        H, W, outH, outW,
        kH, kW, strideH, strideW, padH, padW, dilationH, dilationW,
        channelsPerGroup, outChannelsPerGroup,
        isChannelsLast, dtype,
        bias !== undefined,
        inputStrides, inputOffset
    );

    // Create uniform buffer
    const uniformData = new Int32Array([
        batchSize, inChannels, outChannels,
        H, W, outH, outW,
        kH, kW, strideH, strideW,
        padH, padW, dilationH, dilationW,
        groups, channelsPerGroup, outChannelsPerGroup,
        0  // padding to align
    ]);

    const uniformBuffer = createUniformBuffer(uniformData.buffer);

    // Get or create pipeline (工业级: cache key 包含 strides)
    const cacheKey = `conv-direct-${dtype}-${batchSize}-${inChannels}-${outChannels}-${H}x${W}-k${kH}x${kW}-g${groups}-${isChannelsLast}-${bias ? 'bias' : 'nobias'}-s${inputStrides.join('_')}-o${inputOffset}`;
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
    const weightBuffer = (weight as any).storage.buffer as GPUBuffer;
    const outputBuffer = (config.output as any).storage.buffer as GPUBuffer;

    const entries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: inputBuffer } },
        { binding: 2, resource: { buffer: weightBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
    ];

    if (bias) {
        const biasBuffer = (bias as any).storage.buffer as GPUBuffer;
        entries.push({ binding: 4, resource: { buffer: biasBuffer } });
    }

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries,
    });

    // Dispatch
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

/**
 * 工业级 General Direct Conv Shader 生成 (Strided 支持)
 */
function buildDirectConvShader(
    batchSize: number,
    inChannels: number,
    outChannels: number,
    H: number, W: number,
    outH: number, outW: number,
    kH: number, kW: number,
    strideH: number, strideW: number,
    padH: number, padW: number,
    dilationH: number, dilationW: number,
    channelsPerGroup: number,
    outChannelsPerGroup: number,
    isChannelsLast: boolean,
    dtype: DType,
    hasBias: boolean,
    // 工业级: Strided 支持
    inputStrides: readonly number[],
    inputOffset: number
): string {
    const dataType = dtype === 'float16' ? 'f16' : 'f32';
    const enableF16 = dtype === 'float16' ? 'enable f16;' : '';

    // 工业级: 使用 strided 索引函数
    const inputIndexFn = generateStridedInputIndexFn(isChannelsLast, inputStrides, inputOffset);

    const outputIndexFn = isChannelsLast
        ? `fn getOutputIndex(n: i32, c: i32, h: i32, w: i32) -> i32 {
               return n * (uniforms.outH * uniforms.outW * uniforms.outChannels) +
                      h * (uniforms.outW * uniforms.outChannels) +
                      w * uniforms.outChannels + c;
           }`
        : `fn getOutputIndex(n: i32, c: i32, h: i32, w: i32) -> i32 {
               return n * (uniforms.outChannels * uniforms.outH * uniforms.outW) +
                      c * (uniforms.outH * uniforms.outW) +
                      h * uniforms.outW + w;
           }`;

    const weightIndexFn = `fn getWeightIndex(c_out: i32, c_in_offset: i32, kh: i32, kw: i32) -> i32 {
        return c_out * (uniforms.channelsPerGroup * uniforms.kH * uniforms.kW) +
               c_in_offset * (uniforms.kH * uniforms.kW) +
               kh * uniforms.kW + kw;
    }`;

    const biasBinding = hasBias
        ? `@group(0) @binding(4) var<storage, read> bias: array<${dataType}>;`
        : '';

    const biasAdd = hasBias
        ? 'sum = sum + bias[c_out];'
        : '';

    return `
${enableF16}

${inputIndexFn}

struct Uniforms {
    batchSize: i32,
    inChannels: i32,
    outChannels: i32,
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
    groups: i32,
    channelsPerGroup: i32,
    outChannelsPerGroup: i32,
    _padding: i32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<${dataType}>;
@group(0) @binding(2) var<storage, read> weight: array<${dataType}>;
@group(0) @binding(3) var<storage, read_write> output: array<${dataType}>;
${biasBinding}

${outputIndexFn}
${weightIndexFn}

@compute @workgroup_size(${WORKGROUP_SIZE_X}, ${WORKGROUP_SIZE_Y}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w_out = i32(gid.x);
    let h_out = i32(gid.y);
    let idx = i32(gid.z);
    
    if (w_out >= uniforms.outW || h_out >= uniforms.outH) {
        return;
    }
    
    let n = idx / uniforms.outChannels;
    let c_out = idx % uniforms.outChannels;
    
    if (n >= uniforms.batchSize) {
        return;
    }
    
    // Grouped convolution
    let group = c_out / uniforms.outChannelsPerGroup;
    let c_in_start = group * uniforms.channelsPerGroup;
    
    var sum: ${dataType} = ${dataType}(0.0);
    
    // Convolution inner loops
    for (var c_in_offset: i32 = 0; c_in_offset < uniforms.channelsPerGroup; c_in_offset++) {
        let c_in = c_in_start + c_in_offset;
        
        for (var kh: i32 = 0; kh < uniforms.kH; kh++) {
            for (var kw: i32 = 0; kw < uniforms.kW; kw++) {
                let h_in = h_out * uniforms.strideH - uniforms.padH + kh * uniforms.dilationH;
                let w_in = w_out * uniforms.strideW - uniforms.padW + kw * uniforms.dilationW;
                
                var inputVal: ${dataType} = ${dataType}(0.0);
                if (h_in >= 0 && h_in < uniforms.H && w_in >= 0 && w_in < uniforms.W) {
                    inputVal = input[getInputIndex(n, c_in, h_in, w_in)];
                }
                
                let weightVal = weight[getWeightIndex(c_out, c_in_offset, kh, kw)];
                sum = sum + inputVal * weightVal;
            }
        }
    }
    
    ${biasAdd}
    
    output[getOutputIndex(n, c_out, h_out, w_out)] = sum;
}
`;
}
