/**
 * Winograd Algorithm for 3x3 Convolution - F(4,3)
 * 
 * Winograd F(4,3) 变换用于优化 3x3 卷积。
 * 
 * 算法流程:
 * 1. Filter Transform: G @ filter @ G^T (预计算，只做一次)
 * 2. Input Transform: B^T @ input_tile @ B (每个 6x6 tile)
 * 3. Element-wise Multiply: transformed_input ⊙ transformed_filter
 * 4. Output Transform: A^T @ result @ A (产生 4x4 输出)
 * 
 * 限制条件：
 * - kernel size = 3x3
 * - stride = 1
 * - dilation = 1
 * - groups = 1
 * 
 * 理论加速: 2.25x (减少乘法次数: 36 multiplications -> 16)
 * 
 * **关键修复**: 所有 GPU 操作使用单个 CommandEncoder 提交，
 * 避免异步执行导致的数据同步问题。
 * 
 * @module kernels/window/algorithms/winograd
 */

import type { ITensorHandle, DType } from '@kandle/types';
import type { ConvDispatchResult } from '../types';
import { Logger } from '@kandle/utils';
import { WebGPUDeviceManager } from '../../../base/device';
import { WebGPUPipelineManager } from '../../../pipelines/WebGPUPipelineManager';
import { WebGPUTensor } from '../../../base/tensor';
import { createUniformBuffer as createUniformBufferFromPool } from '../../../base/uniformUtils';

const logger = new Logger('Winograd');

// ============================================================================
// Winograd F(4,3) Constants
// ============================================================================

/** 输出 tile 大小 */
const TILE_M = 4;
/** Filter 大小 */
const TILE_R = 3;
/** 输入 tile 大小 = TILE_M + TILE_R - 1 = 6 */
const TILE_SIZE = TILE_M + TILE_R - 1; // 6

/** Workgroup 配置 */
const WORKGROUP_SIZE = 64;

// ============================================================================
// Winograd F(4,3) Transform Matrices (Float32 for precision)
// ============================================================================

/**
 * B^T: 输入变换矩阵 (转置形式，6x6)
 * 
 * 用于: d = B^T @ input_tile @ B
 */
const Bt: number[][] = [
    [4, 0, -5, 0, 1, 0],
    [0, -4, -4, 1, 1, 0],
    [0, 4, -4, -1, 1, 0],
    [0, -2, -1, 2, 1, 0],
    [0, 2, -1, -2, 1, 0],
    [0, 4, 0, -5, 0, 1],
];

/**
 * B: 输入变换矩阵 (6x6)
 */
const B: number[][] = transposeMatrix(Bt);

/**
 * G: 权重变换矩阵 (6x3)
 * 
 * 用于: u = G @ filter @ G^T
 */
const G: number[][] = [
    [1 / 4, 0, 0],
    [-1 / 6, -1 / 6, -1 / 6],
    [-1 / 6, 1 / 6, -1 / 6],
    [1 / 24, 1 / 12, 1 / 6],
    [1 / 24, -1 / 12, 1 / 6],
    [0, 0, 1],
];

/**
 * G^T: 权重变换矩阵的转置 (3x6)
 */
const Gt: number[][] = transposeMatrix(G);

/**
 * A^T: 输出变换矩阵 (转置形式，4x6)
 * 
 * 用于: output = A^T @ (d ⊙ u) @ A
 */
const At: number[][] = [
    [1, 1, 1, 1, 1, 0],
    [0, 1, -1, 2, -2, 0],
    [0, 1, 1, 4, 4, 0],
    [0, 1, -1, 8, -8, 1],
];

/**
 * A: 输出变换矩阵 (6x4)
 */
const A: number[][] = transposeMatrix(At);

// ============================================================================
// Matrix Utilities
// ============================================================================

function transposeMatrix(m: number[][]): number[][] {
    const rows = m.length;
    const cols = m[0].length;
    const result: number[][] = [];
    for (let j = 0; j < cols; j++) {
        result[j] = [];
        for (let i = 0; i < rows; i++) {
            result[j][i] = m[i][j];
        }
    }
    return result;
}

function flattenMatrix(m: number[][]): Float32Array {
    const rows = m.length;
    const cols = m[0].length;
    const data: number[] = [];
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            data.push(m[i][j]);
        }
    }
    return new Float32Array(data);
}

// ============================================================================
// Pipeline Context - 用于收集所有 passes 然后一次性提交
// ============================================================================

interface WinogradPassConfig {
    pipeline: GPUComputePipeline;
    bindGroup: GPUBindGroup;
    workgroups: [number, number, number];
}

/**
 * Winograd 执行上下文
 * 收集所有 compute passes，统一在一个 CommandEncoder 中执行
 */
class WinogradExecutionContext {
    private passes: WinogradPassConfig[] = [];
    private device: GPUDevice;
    private uniformBuffers: GPUBuffer[] = [];
    private matrixBuffers: GPUBuffer[] = [];

    constructor(device: GPUDevice) {
        this.device = device;
    }

    addPass(config: WinogradPassConfig): void {
        this.passes.push(config);
    }

    /** 创建 uniform buffer 并记录以便管理 */
    createUniformBuffer(data: Int32Array | Float32Array): GPUBuffer {
        const buffer = createUniformBufferFromPool(data.buffer as ArrayBuffer);
        this.uniformBuffers.push(buffer);
        return buffer;
    }

    /** 创建 storage buffer 用于变换矩阵 */
    createMatrixBuffer(data: Float32Array): GPUBuffer {
        const buffer = this.device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
        this.matrixBuffers.push(buffer);
        return buffer;
    }

    /** 执行所有收集的 passes，一次性提交 */
    execute(): void {
        const commandEncoder = this.device.createCommandEncoder();

        for (const pass of this.passes) {
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(pass.pipeline);
            passEncoder.setBindGroup(0, pass.bindGroup);
            passEncoder.dispatchWorkgroups(...pass.workgroups);
            passEncoder.end();
        }

        this.device.queue.submit([commandEncoder.finish()]);
    }
}

// ============================================================================
// Main Execute Function
// ============================================================================

/**
 * 执行 Winograd F(4,3) 卷积
 * 
 * **关键**: 所有 GPU 操作收集后统一提交，确保正确的执行顺序
 * 
 * @param config - Conv dispatch 配置
 * @param input - 输入张量 [N, C_in, H, W] 或 [N, H, W, C_in]
 * @param weight - 权重张量 [C_out, C_in, 3, 3]
 * @param bias - 可选偏置 [C_out]
 */
export function executeWinogradConv(
    config: ConvDispatchResult,
    input: ITensorHandle,
    weight: ITensorHandle,
    bias?: ITensorHandle
): void {
    // Validate constraints
    validateWinogradConstraints(config);

    const device = WebGPUDeviceManager.device;
    const { batchSize, inChannels, outChannels } = config;
    const [H, W] = config.inputSpatial;
    const [outH, outW] = config.outputSpatial;
    const [padH, padW] = config.padding;
    const isChannelsLast = config.isChannelsLast;
    const dtype = config.computeDtype;

    // 计算 tile 数量
    const numTilesH = Math.ceil(outH / TILE_M);
    const numTilesW = Math.ceil(outW / TILE_M);
    const numTiles = numTilesH * numTilesW;

    logger.debug(`Winograd F(4,3): N=${batchSize}, C_in=${inChannels}, C_out=${outChannels}`);
    logger.debug(`  Input: ${H}x${W}, Output: ${outH}x${outW}`);
    logger.debug(`  Tiles: ${numTilesH}x${numTilesW} = ${numTiles}`);

    // 创建执行上下文
    const ctx = new WinogradExecutionContext(device);

    // 分配中间缓冲区
    const transformedWeight = createTransformBuffer(outChannels * inChannels * TILE_SIZE * TILE_SIZE, dtype);
    const transformedInputSize = batchSize * numTiles * inChannels * TILE_SIZE * TILE_SIZE;
    const transformedInput = createTransformBuffer(transformedInputSize, dtype);
    const batchedProductSize = batchSize * numTiles * outChannels * TILE_SIZE * TILE_SIZE;
    const batchedProduct = createTransformBuffer(batchedProductSize, dtype);

    // 步骤 1: 准备权重变换 pass
    prepareFilterTransformPass(ctx, device, weight, transformedWeight, inChannels, outChannels, dtype);

    // 步骤 2: 准备输入变换 pass
    prepareInputTransformPass(
        ctx, device, input, transformedInput,
        batchSize, inChannels, H, W,
        numTilesH, numTilesW, padH, padW,
        isChannelsLast, dtype
    );

    // 步骤 3: 准备批量乘法 pass
    prepareBatchedMulPass(
        ctx, device, transformedInput, transformedWeight, batchedProduct,
        batchSize, numTiles, inChannels, outChannels, dtype
    );

    // 步骤 4: 准备输出变换 pass
    prepareOutputTransformPass(
        ctx, device, batchedProduct, config.output, bias,
        batchSize, outChannels, outH, outW,
        numTilesH, numTilesW, isChannelsLast, dtype
    );

    // 一次性执行所有 passes
    ctx.execute();
}

// ============================================================================
// Constraint Validation
// ============================================================================

function validateWinogradConstraints(config: ConvDispatchResult): void {
    const [kH, kW] = config.kernelSize;
    const [strideH, strideW] = config.stride;
    const [dilationH, dilationW] = config.dilation;

    if (kH !== 3 || kW !== 3) {
        throw new Error(`Winograd F(4,3) only supports 3x3 kernels, got ${kH}x${kW}`);
    }
    if (strideH !== 1 || strideW !== 1) {
        throw new Error(`Winograd F(4,3) only supports stride=1, got ${strideH}x${strideW}`);
    }
    if (dilationH !== 1 || dilationW !== 1) {
        throw new Error(`Winograd F(4,3) only supports dilation=1, got ${dilationH}x${dilationW}`);
    }
    if (config.groups !== 1) {
        throw new Error(`Winograd F(4,3) only supports groups=1, got ${config.groups}`);
    }
}

/**
 * 检查是否可以使用 Winograd 算法
 */
export function canUseWinograd(config: ConvDispatchResult): boolean {
    const [kH, kW] = config.kernelSize;
    const [strideH, strideW] = config.stride;
    const [dilationH, dilationW] = config.dilation;

    return (
        kH === 3 && kW === 3 &&
        strideH === 1 && strideW === 1 &&
        dilationH === 1 && dilationW === 1 &&
        config.groups === 1
    );
}

// ============================================================================
// Buffer Creation
// ============================================================================

function createTransformBuffer(size: number, dtype: DType): ITensorHandle {
    return WebGPUTensor.createNew([size], dtype);
}

// ============================================================================
// Filter Transform Pass: u = G @ filter @ G^T
// ============================================================================

function prepareFilterTransformPass(
    ctx: WinogradExecutionContext,
    device: GPUDevice,
    weight: ITensorHandle,
    output: ITensorHandle,
    inChannels: number,
    outChannels: number,
    dtype: DType
): void {
    const shaderCode = buildFilterTransformShader(inChannels, outChannels, dtype);

    const uniformBuffer = ctx.createUniformBuffer(
        new Int32Array([inChannels, outChannels, TILE_R, TILE_SIZE])
    );
    const gBuffer = ctx.createMatrixBuffer(flattenMatrix(G));
    const gtBuffer = ctx.createMatrixBuffer(flattenMatrix(Gt));

    const cacheKey = `winograd-filter-transform-${dtype}-${inChannels}-${outChannels}`;
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

    const weightBuffer = (weight as any).storage.buffer as GPUBuffer;
    const outputBuffer = (output as any).storage.buffer as GPUBuffer;

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: weightBuffer } },
            { binding: 2, resource: { buffer: gBuffer } },
            { binding: 3, resource: { buffer: gtBuffer } },
            { binding: 4, resource: { buffer: outputBuffer } },
        ],
    });

    const totalFilters = outChannels * inChannels;
    const workgroups = Math.ceil(totalFilters / WORKGROUP_SIZE);

    ctx.addPass({
        pipeline,
        bindGroup,
        workgroups: [workgroups, 1, 1],
    });
}

function buildFilterTransformShader(inChannels: number, outChannels: number, dtype: DType): string {
    const dataType = dtype === 'float16' ? 'f16' : 'f32';
    const enableF16 = dtype === 'float16' ? 'enable f16;' : '';
    const totalFilters = outChannels * inChannels;

    return `
${enableF16}

struct Uniforms {
    inChannels: i32,
    outChannels: i32,
    filterSize: i32,  // 3
    tileSize: i32,    // 6
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> weight: array<${dataType}>;
@group(0) @binding(2) var<storage, read> G: array<f32>;      // 6x3
@group(0) @binding(3) var<storage, read> Gt: array<f32>;     // 3x6
@group(0) @binding(4) var<storage, read_write> output: array<${dataType}>;

// G @ filter @ G^T
// G: 6x3, filter: 3x3, Gt: 3x6
// temp = G @ filter: 6x3
// output = temp @ Gt: 6x6

@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = i32(gid.x);
    if (idx >= ${totalFilters}) {
        return;
    }
    
    let c_out = idx / uniforms.inChannels;
    let c_in = idx % uniforms.inChannels;
    
    // 读取 3x3 kernel weights
    var kernelData: array<${dataType}, 9>;
    let kernelOffset = c_out * uniforms.inChannels * 9 + c_in * 9;
    for (var i: i32 = 0; i < 9; i++) {
        kernelData[i] = weight[kernelOffset + i];
    }
    
    // temp = G @ kernel (6x3 @ 3x3 = 6x3)
    var temp: array<${dataType}, 18>;  // 6x3
    for (var i: i32 = 0; i < 6; i++) {
        for (var j: i32 = 0; j < 3; j++) {
            var sum: ${dataType} = ${dataType}(0.0);
            for (var k: i32 = 0; k < 3; k++) {
                sum = sum + ${dataType}(G[i * 3 + k]) * kernelData[k * 3 + j];
            }
            temp[i * 3 + j] = sum;
        }
    }
    
    // output = temp @ Gt (6x3 @ 3x6 = 6x6)
    let outputOffset = idx * 36;  // 6x6 = 36
    for (var i: i32 = 0; i < 6; i++) {
        for (var j: i32 = 0; j < 6; j++) {
            var sum: ${dataType} = ${dataType}(0.0);
            for (var k: i32 = 0; k < 3; k++) {
                sum = sum + temp[i * 3 + k] * ${dataType}(Gt[k * 6 + j]);
            }
            output[outputOffset + i * 6 + j] = sum;
        }
    }
}
`;
}

// ============================================================================
// Input Transform Pass: d = B^T @ input_tile @ B
// ============================================================================

function prepareInputTransformPass(
    ctx: WinogradExecutionContext,
    device: GPUDevice,
    input: ITensorHandle,
    output: ITensorHandle,
    batchSize: number,
    channels: number,
    H: number, W: number,
    numTilesH: number, numTilesW: number,
    padH: number, padW: number,
    isChannelsLast: boolean,
    dtype: DType
): void {
    const shaderCode = buildInputTransformShader(
        batchSize, channels, H, W,
        numTilesH, numTilesW, padH, padW,
        isChannelsLast, dtype
    );

    const uniformBuffer = ctx.createUniformBuffer(new Int32Array([
        batchSize, channels, H, W,
        numTilesH, numTilesW, padH, padW,
        TILE_M, TILE_SIZE, 0, 0  // padding for alignment
    ]));
    const btBuffer = ctx.createMatrixBuffer(flattenMatrix(Bt));
    const bBuffer = ctx.createMatrixBuffer(flattenMatrix(B));

    const numTiles = numTilesH * numTilesW;
    const totalWork = batchSize * numTiles * channels;

    const cacheKey = `winograd-input-transform-${dtype}-${batchSize}-${channels}-${H}x${W}-${numTilesH}x${numTilesW}-${isChannelsLast}`;
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
            { binding: 2, resource: { buffer: btBuffer } },
            { binding: 3, resource: { buffer: bBuffer } },
            { binding: 4, resource: { buffer: outputBuffer } },
        ],
    });

    const workgroups = Math.ceil(totalWork / WORKGROUP_SIZE);

    ctx.addPass({
        pipeline,
        bindGroup,
        workgroups: [workgroups, 1, 1],
    });
}

function buildInputTransformShader(
    batchSize: number,
    channels: number,
    H: number, W: number,
    _numTilesH: number, _numTilesW: number,
    _padH: number, _padW: number,
    isChannelsLast: boolean,
    dtype: DType
): string {
    const dataType = dtype === 'float16' ? 'f16' : 'f32';
    const enableF16 = dtype === 'float16' ? 'enable f16;' : '';

    const getInputIndex = isChannelsLast
        ? `fn getInputIndex(n: i32, c: i32, h: i32, w: i32) -> i32 {
               return n * (uniforms.H * uniforms.W * uniforms.channels) +
                      h * (uniforms.W * uniforms.channels) +
                      w * uniforms.channels + c;
           }`
        : `fn getInputIndex(n: i32, c: i32, h: i32, w: i32) -> i32 {
               return n * (uniforms.channels * uniforms.H * uniforms.W) +
                      c * (uniforms.H * uniforms.W) +
                      h * uniforms.W + w;
           }`;

    return `
${enableF16}

struct Uniforms {
    batchSize: i32,
    channels: i32,
    H: i32,
    W: i32,
    numTilesH: i32,
    numTilesW: i32,
    padH: i32,
    padW: i32,
    tileM: i32,      // 4
    tileSize: i32,   // 6
    _pad1: i32,
    _pad2: i32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<${dataType}>;
@group(0) @binding(2) var<storage, read> Bt: array<f32>;  // 6x6
@group(0) @binding(3) var<storage, read> B: array<f32>;   // 6x6
@group(0) @binding(4) var<storage, read_write> output: array<${dataType}>;

${getInputIndex}

fn getTile(n: i32, c: i32, tile_h: i32, tile_w: i32, local_h: i32, local_w: i32) -> ${dataType} {
    // 计算在原始输入中的位置
    let h = tile_h * uniforms.tileM + local_h - uniforms.padH;
    let w = tile_w * uniforms.tileM + local_w - uniforms.padW;
    
    if (h < 0 || h >= uniforms.H || w < 0 || w >= uniforms.W) {
        return ${dataType}(0.0);
    }
    
    return input[getInputIndex(n, c, h, w)];
}

@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = i32(gid.x);
    let numTiles = uniforms.numTilesH * uniforms.numTilesW;
    let totalWork = uniforms.batchSize * numTiles * uniforms.channels;
    
    if (idx >= totalWork) {
        return;
    }
    
    // Decode index -> (n, tile_idx, c)
    let c = idx % uniforms.channels;
    var temp = idx / uniforms.channels;
    let tile_idx = temp % numTiles;
    let n = temp / numTiles;
    
    let tile_h = tile_idx / uniforms.numTilesW;
    let tile_w = tile_idx % uniforms.numTilesW;
    
    // 读取 6x6 input tile
    var inputTile: array<${dataType}, 36>;
    for (var i: i32 = 0; i < 6; i++) {
        for (var j: i32 = 0; j < 6; j++) {
            inputTile[i * 6 + j] = getTile(n, c, tile_h, tile_w, i, j);
        }
    }
    
    // temp = Bt @ inputTile (6x6 @ 6x6 = 6x6)
    var tempResult: array<${dataType}, 36>;
    for (var i: i32 = 0; i < 6; i++) {
        for (var j: i32 = 0; j < 6; j++) {
            var sum: ${dataType} = ${dataType}(0.0);
            for (var k: i32 = 0; k < 6; k++) {
                sum = sum + ${dataType}(Bt[i * 6 + k]) * inputTile[k * 6 + j];
            }
            tempResult[i * 6 + j] = sum;
        }
    }
    
    // output = temp @ B (6x6 @ 6x6 = 6x6)
    // Output layout: [N, numTiles, C, 6, 6]
    let outputOffset = ((n * numTiles + tile_idx) * uniforms.channels + c) * 36;
    for (var i: i32 = 0; i < 6; i++) {
        for (var j: i32 = 0; j < 6; j++) {
            var sum: ${dataType} = ${dataType}(0.0);
            for (var k: i32 = 0; k < 6; k++) {
                sum = sum + tempResult[i * 6 + k] * ${dataType}(B[k * 6 + j]);
            }
            output[outputOffset + i * 6 + j] = sum;
        }
    }
}
`;
}

// ============================================================================
// Batched Multiply Pass: Element-wise multiply in Winograd domain
// ============================================================================

function prepareBatchedMulPass(
    ctx: WinogradExecutionContext,
    device: GPUDevice,
    transformedInput: ITensorHandle,
    transformedWeight: ITensorHandle,
    output: ITensorHandle,
    batchSize: number,
    numTiles: number,
    inChannels: number,
    outChannels: number,
    dtype: DType
): void {
    const shaderCode = buildBatchedMulShader(batchSize, numTiles, inChannels, outChannels, dtype);

    const uniformBuffer = ctx.createUniformBuffer(
        new Int32Array([batchSize, numTiles, inChannels, outChannels])
    );

    const totalWork = batchSize * numTiles * outChannels * TILE_SIZE * TILE_SIZE;

    const cacheKey = `winograd-batched-mul-${dtype}-${batchSize}-${numTiles}-${inChannels}-${outChannels}`;
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

    const inputBuffer = (transformedInput as any).storage.buffer as GPUBuffer;
    const weightBuffer = (transformedWeight as any).storage.buffer as GPUBuffer;
    const outputBuffer = (output as any).storage.buffer as GPUBuffer;

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: inputBuffer } },
            { binding: 2, resource: { buffer: weightBuffer } },
            { binding: 3, resource: { buffer: outputBuffer } },
        ],
    });

    const workgroups = Math.ceil(totalWork / WORKGROUP_SIZE);

    ctx.addPass({
        pipeline,
        bindGroup,
        workgroups: [workgroups, 1, 1],
    });
}

function buildBatchedMulShader(
    batchSize: number,
    numTiles: number,
    inChannels: number,
    outChannels: number,
    dtype: DType
): string {
    const dataType = dtype === 'float16' ? 'f16' : 'f32';
    const enableF16 = dtype === 'float16' ? 'enable f16;' : '';
    const totalWork = batchSize * numTiles * outChannels * TILE_SIZE * TILE_SIZE;

    return `
${enableF16}

struct Uniforms {
    batchSize: i32,
    numTiles: i32,
    inChannels: i32,
    outChannels: i32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> transformedInput: array<${dataType}>;
@group(0) @binding(2) var<storage, read> transformedWeight: array<${dataType}>;
@group(0) @binding(3) var<storage, read_write> output: array<${dataType}>;

@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = i32(gid.x);
    if (idx >= ${totalWork}) {
        return;
    }
    
    // Decode index -> (n, tile, c_out, pos)
    let pos = idx % 36;  // 6x6 = 36
    var temp = idx / 36;
    let c_out = temp % uniforms.outChannels;
    temp = temp / uniforms.outChannels;
    let tile = temp % uniforms.numTiles;
    let n = temp / uniforms.numTiles;
    
    var sum: ${dataType} = ${dataType}(0.0);
    
    // Sum over input channels
    for (var c_in: i32 = 0; c_in < uniforms.inChannels; c_in++) {
        // Input index: [N, numTiles, C_in, 36]
        let inputIdx = ((n * uniforms.numTiles + tile) * uniforms.inChannels + c_in) * 36 + pos;
        // Weight index: [C_out, C_in, 36]
        let weightIdx = (c_out * uniforms.inChannels + c_in) * 36 + pos;
        
        sum = sum + transformedInput[inputIdx] * transformedWeight[weightIdx];
    }
    
    // Output index: [N, numTiles, C_out, 36]
    let outputIdx = ((n * uniforms.numTiles + tile) * uniforms.outChannels + c_out) * 36 + pos;
    output[outputIdx] = sum;
}
`;
}

// ============================================================================
// Output Transform Pass: output = A^T @ product @ A + bias
// ============================================================================

function prepareOutputTransformPass(
    ctx: WinogradExecutionContext,
    device: GPUDevice,
    product: ITensorHandle,
    output: ITensorHandle,
    bias: ITensorHandle | undefined,
    batchSize: number,
    outChannels: number,
    outH: number, outW: number,
    numTilesH: number, numTilesW: number,
    isChannelsLast: boolean,
    dtype: DType
): void {
    const shaderCode = buildOutputTransformShader(
        batchSize, outChannels, outH, outW,
        numTilesH, numTilesW, isChannelsLast, dtype,
        bias !== undefined
    );

    const uniformBuffer = ctx.createUniformBuffer(new Int32Array([
        batchSize, outChannels, outH, outW,
        numTilesH, numTilesW, TILE_M, TILE_SIZE
    ]));
    const atBuffer = ctx.createMatrixBuffer(flattenMatrix(At));
    const aBuffer = ctx.createMatrixBuffer(flattenMatrix(A));

    const numTiles = numTilesH * numTilesW;
    const totalWork = batchSize * numTiles * outChannels;

    const cacheKey = `winograd-output-transform-${dtype}-${batchSize}-${outChannels}-${outH}x${outW}-${isChannelsLast}-${bias !== undefined}`;
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

    const productBuffer = (product as any).storage.buffer as GPUBuffer;
    const outputBuffer = (output as any).storage.buffer as GPUBuffer;

    const entries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: productBuffer } },
        { binding: 2, resource: { buffer: atBuffer } },
        { binding: 3, resource: { buffer: aBuffer } },
        { binding: 4, resource: { buffer: outputBuffer } },
    ];

    if (bias) {
        const biasBuffer = (bias as any).storage.buffer as GPUBuffer;
        entries.push({ binding: 5, resource: { buffer: biasBuffer } });
    }

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries,
    });

    const workgroups = Math.ceil(totalWork / WORKGROUP_SIZE);

    ctx.addPass({
        pipeline,
        bindGroup,
        workgroups: [workgroups, 1, 1],
    });
}

function buildOutputTransformShader(
    _batchSize: number,
    _outChannels: number,
    _outH: number, _outW: number,
    _numTilesH: number, _numTilesW: number,
    isChannelsLast: boolean,
    dtype: DType,
    hasBias: boolean
): string {
    const dataType = dtype === 'float16' ? 'f16' : 'f32';
    const enableF16 = dtype === 'float16' ? 'enable f16;' : '';

    const getOutputIndex = isChannelsLast
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

    const biasBinding = hasBias
        ? `@group(0) @binding(5) var<storage, read> bias: array<${dataType}>;`
        : '';

    const biasAdd = hasBias
        ? 'result[i * 4 + j] = result[i * 4 + j] + bias[c_out];'
        : '';

    return `
${enableF16}

struct Uniforms {
    batchSize: i32,
    outChannels: i32,
    outH: i32,
    outW: i32,
    numTilesH: i32,
    numTilesW: i32,
    tileM: i32,      // 4
    tileSize: i32,   // 6
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> product: array<${dataType}>;
@group(0) @binding(2) var<storage, read> At: array<f32>;  // 4x6
@group(0) @binding(3) var<storage, read> A: array<f32>;   // 6x4
@group(0) @binding(4) var<storage, read_write> output: array<${dataType}>;
${biasBinding}

${getOutputIndex}

@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = i32(gid.x);
    let numTiles = uniforms.numTilesH * uniforms.numTilesW;
    let totalWork = uniforms.batchSize * numTiles * uniforms.outChannels;
    
    if (idx >= totalWork) {
        return;
    }
    
    // Decode index -> (n, tile, c_out)
    let c_out = idx % uniforms.outChannels;
    var temp = idx / uniforms.outChannels;
    let tile_idx = temp % numTiles;
    let n = temp / numTiles;
    
    let tile_h = tile_idx / uniforms.numTilesW;
    let tile_w = tile_idx % uniforms.numTilesW;
    
    // 读取 6x6 product 数据
    // Product layout: [N, numTiles, C_out, 36]
    let productOffset = ((n * numTiles + tile_idx) * uniforms.outChannels + c_out) * 36;
    var productTile: array<${dataType}, 36>;
    for (var i: i32 = 0; i < 36; i++) {
        productTile[i] = product[productOffset + i];
    }
    
    // temp = At @ productTile (4x6 @ 6x6 = 4x6)
    var tempResult: array<${dataType}, 24>;  // 4x6
    for (var i: i32 = 0; i < 4; i++) {
        for (var j: i32 = 0; j < 6; j++) {
            var sum: ${dataType} = ${dataType}(0.0);
            for (var k: i32 = 0; k < 6; k++) {
                sum = sum + ${dataType}(At[i * 6 + k]) * productTile[k * 6 + j];
            }
            tempResult[i * 6 + j] = sum;
        }
    }
    
    // result = temp @ A (4x6 @ 6x4 = 4x4)
    var result: array<${dataType}, 16>;
    for (var i: i32 = 0; i < 4; i++) {
        for (var j: i32 = 0; j < 4; j++) {
            var sum: ${dataType} = ${dataType}(0.0);
            for (var k: i32 = 0; k < 6; k++) {
                sum = sum + tempResult[i * 6 + k] * ${dataType}(A[k * 4 + j]);
            }
            result[i * 4 + j] = sum;
            ${biasAdd}
        }
    }
    
    // 写入输出，处理边界情况
    let baseH = tile_h * uniforms.tileM;
    let baseW = tile_w * uniforms.tileM;
    
    for (var i: i32 = 0; i < 4; i++) {
        for (var j: i32 = 0; j < 4; j++) {
            let oh = baseH + i;
            let ow = baseW + j;
            
            if (oh < uniforms.outH && ow < uniforms.outW) {
                output[getOutputIndex(n, c_out, oh, ow)] = result[i * 4 + j];
            }
        }
    }
}
`;
}
