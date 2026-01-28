/**
 * IIR Filter Executor
 *
 * 执行 IIR 滤波器的 WebGPU kernel
 *
 * 策略:
 * 每个线程处理一个 batch 项，时间维度串行处理
 * 保证 IIR 递归正确性，同时利用 batch 并行
 */

import type { ITensorHandle, DType } from '@kandle/types';
import { Logger } from '@kandle/utils';
import { WebGPUDeviceManager } from '../../../base/device';
import { WebGPUPipelineManager } from '../../../pipelines/WebGPUPipelineManager';
import { createUniformBuffer } from '../../../base/uniformUtils';
import type { WebGPUTensor } from '../../../base/tensor';
import { BiquadCoeffs, IIR_WORKGROUP_SIZE } from './types';

const logger = new Logger('IIR-Executor');

/**
 * IIR Biquad Kernel Args (与 handler 中的定义对应)
 */
export interface IIRBiquadKernelArgs {
    input: ITensorHandle;
    output: ITensorHandle;
    b0: number;
    b1: number;
    b2: number;
    a1: number;
    a2: number;
    clamp: boolean;
    clampMin: number;
    clampMax: number;
}

/**
 * Kernel 执行上下文 (简化版，避免跨包依赖)
 */
interface KernelContext {
    scalars: Record<string, unknown>;
}

/**
 * 注册 IIR kernel
 */
export function registerIIRkernel(registry: { register: (name: string, impl: (ctx: KernelContext) => void) => void }): void {
    registry.register('iir.biquad', executeIIRBiquadKernel);
}

/**
 * IIR Biquad Kernel 入口
 *
 * 从 KernelContext 提取参数并执行
 */
function executeIIRBiquadKernel(ctx: KernelContext): void {
    const args = ctx.scalars as unknown as IIRBiquadKernelArgs;
    const input = args.input as WebGPUTensor<DType>;
    const output = args.output as WebGPUTensor<DType>;

    // 从 tensor handle 获取 shape 信息
    const shape = input.shape;
    const ndim = shape.length;
    const signalLength = shape[ndim - 1];

    // 计算 batch 大小 (所有非最后一维的乘积)
    const batchDims = shape.slice(0, -1);
    const batchSize = batchDims.reduce((a: number, b: number) => a * b, 1);

    // 获取 GPU buffer
    const inputBuffer = input.storage.buffer as GPUBuffer;
    const outputBuffer = output.storage.buffer as GPUBuffer;

    const coeffs: BiquadCoeffs = {
        b0: args.b0,
        b1: args.b1,
        b2: args.b2,
        a1: args.a1,
        a2: args.a2,
    };

    executeIIRFilter(
        inputBuffer,
        outputBuffer,
        input.offset,
        output.offset,
        signalLength,
        batchSize,
        coeffs,
        args.clamp,
        args.clampMin,
        args.clampMax
    );
}

/**
 * 执行 IIR 滤波
 */
function executeIIRFilter(
    inputBuffer: GPUBuffer,
    outputBuffer: GPUBuffer,
    inputOffset: number,
    outputOffset: number,
    signalLength: number,
    batchSize: number,
    coeffs: BiquadCoeffs,
    clamp: boolean = true,
    clampMin: number = -1.0,
    clampMax: number = 1.0
): void {
    const device = WebGPUDeviceManager.device;

    logger.debug(`IIR Filter: signal_len=${signalLength}, batch=${batchSize}, clamp=${clamp}`);

    // =========================================================================
    // 生成 shader
    // =========================================================================

    const shaderCode = buildIIRShader(signalLength, batchSize, coeffs, clamp, clampMin, clampMax);

    // Pipeline key
    const pipelineKey = `iir_filter-s${signalLength}-b${batchSize}-c${clamp}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const module = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
        logger.debug(`Created IIR pipeline: ${pipelineKey}`);
    }

    // =========================================================================
    // 创建 uniform buffer
    // =========================================================================

    const uniformSize = 48; // 12 * 4 bytes
    const uniformData = new ArrayBuffer(uniformSize);
    const uniformView = new DataView(uniformData);

    let offset = 0;
    uniformView.setUint32(offset, signalLength, true); offset += 4;
    uniformView.setUint32(offset, batchSize, true); offset += 4;
    uniformView.setFloat32(offset, coeffs.b0, true); offset += 4;
    uniformView.setFloat32(offset, coeffs.b1, true); offset += 4;
    uniformView.setFloat32(offset, coeffs.b2, true); offset += 4;
    uniformView.setFloat32(offset, coeffs.a1, true); offset += 4;
    uniformView.setFloat32(offset, coeffs.a2, true); offset += 4;
    uniformView.setUint32(offset, inputOffset, true); offset += 4;
    uniformView.setUint32(offset, outputOffset, true); offset += 4;
    uniformView.setFloat32(offset, clampMin, true); offset += 4;
    uniformView.setFloat32(offset, clampMax, true); offset += 4;
    uniformView.setUint32(offset, 0, true); // _pad

    const uniformBuffer = createUniformBuffer(uniformData);

    // =========================================================================
    // 创建 bind group 和执行
    // =========================================================================

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: inputBuffer } },
            { binding: 2, resource: { buffer: outputBuffer } },
        ],
    });

    // 每个 workgroup 处理多个 batch 项
    const numWorkgroups = Math.ceil(batchSize / IIR_WORKGROUP_SIZE);

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    logger.debug(`IIR Filter complete: ${numWorkgroups} workgroups`);
}

/**
 * 构建 IIR shader
 *
 * 策略: 每个线程处理一个完整的 batch 项 (串行遍历时间维度)
 * 这保证了 IIR 递归的正确性, 同时利用 batch 并行
 */
function buildIIRShader(
    signalLength: number,
    batchSize: number,
    coeffs: BiquadCoeffs,
    clamp: boolean,
    clampMin: number,
    clampMax: number
): string {
    const clampCode = clamp
        ? `y_n = clamp(y_n, uniforms.clamp_min, uniforms.clamp_max);`
        : '';

    return `
// IIR Biquad Filter Shader
// Each thread processes one batch item sequentially in time
// Signal length: ${signalLength}, Batch size: ${batchSize}

struct Uniforms {
    signal_length: u32,
    batch_size: u32,
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    input_offset: u32,
    output_offset: u32,
    clamp_min: f32,
    clamp_max: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(${IIR_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.x;

    if (batch_idx >= uniforms.batch_size) {
        return;
    }

    let base_offset = batch_idx * uniforms.signal_length;

    // IIR 状态变量
    var y_n1: f32 = 0.0;  // y[n-1]
    var y_n2: f32 = 0.0;  // y[n-2]
    var x_n1: f32 = 0.0;  // x[n-1]
    var x_n2: f32 = 0.0;  // x[n-2]

    // 串行处理时间维度
    for (var n = 0u; n < uniforms.signal_length; n++) {
        let x_n = input[uniforms.input_offset + base_offset + n];

        // IIR 差分方程:
        // y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        var y_n = uniforms.b0 * x_n
                + uniforms.b1 * x_n1
                + uniforms.b2 * x_n2
                - uniforms.a1 * y_n1
                - uniforms.a2 * y_n2;

        // 可选 clamp
        ${clampCode}

        // 写入输出
        output[uniforms.output_offset + base_offset + n] = y_n;

        // 更新状态
        x_n2 = x_n1;
        x_n1 = x_n;
        y_n2 = y_n1;
        y_n1 = y_n;
    }
}
`;
}
