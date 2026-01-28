/**
 * Diff Kernel Executor
 * 
 * Executes N-order forward difference using WebGPU compute shaders.
 * 
 * diff(input, n=1, dim=-1):
 *   out[i] = input[i+1] - input[i]  (1st order)
 *   Higher orders: apply 1st order recursively
 */

import type { ITensorHandle, DType } from '@kandle/types';
import { WebGPUDeviceManager } from '../../../base/device';
import { WebGPUPipelineManager } from '../../../pipelines/WebGPUPipelineManager';
import { WebGPUTensor } from '../../../base/tensor';
import { Logger } from '@kandle/utils';
import { buildDiffShader } from './shaderBuilder';
import type { DiffParams } from './types';
import { createUniformBuffer as createUniformBufferFromPool } from '../../../base/uniformUtils';
import { getGlobalDTypeResolver } from '../../../base/DTypeResolver';

const logger = new Logger('Diff-Executor');

const MAX_RANK = 8;

/**
 * Uniform buffer size
 * Header: 8 * 4 = 32 bytes
 * input_shape: 8 * 4 = 32 bytes
 * input_strides: 8 * 4 = 32 bytes
 * output_shape: 8 * 4 = 32 bytes
 * output_strides: 8 * 4 = 32 bytes
 * Total: 160 bytes
 */
const UNIFORM_BUFFER_SIZE = 160;

/**
 * Diff kernel implementation
 * 
 * Signature: (input, scalars, outs?) => ITensorHandle
 */
export function diffKernel(
    input: ITensorHandle,
    scalars: Record<string, unknown>,
    outs?: ITensorHandle[]
): ITensorHandle {
    const device = WebGPUDeviceManager.device;
    const resolver = getGlobalDTypeResolver();

    // 1. Get scalar args
    const n = (scalars['n'] ?? 1) as number;
    let dim = (scalars['dim'] ?? -1) as number;

    // 2. Compute dimensions
    const inputShape = [...input.shape] as number[];
    const inputStrides = [...input.strides] as number[];
    const rank = inputShape.length;
    const dtype = input.dtype;

    // Normalize dim
    if (dim < 0) {
        dim = dim + rank;
    }

    // 3. Validate
    if (n < 1) {
        throw new Error(`diff: n must be >= 1, got ${n}`);
    }
    if (dim < 0 || dim >= rank) {
        throw new Error(`diff: dim ${dim} out of range for tensor of rank ${rank}`);
    }
    if (inputShape[dim] <= n) {
        throw new Error(`diff: input size along dim ${dim} is ${inputShape[dim]}, must be > n=${n}`);
    }

    // 4. Calculate output shape (size along dim reduced by n)
    const outputShape = [...inputShape];
    outputShape[dim] = inputShape[dim] - n;

    // 5. Allocate output tensor
    const output = outs?.[0] as WebGPUTensor<typeof dtype>
        ?? WebGPUTensor.createNew(outputShape as number[], dtype as DType);

    const outputNumel = outputShape.reduce((a, b) => a * b, 1);

    // For n > 1, we need to apply diff recursively
    // For now, implement only n=1 directly; higher orders can be composite
    if (n !== 1) {
        // Recursive implementation: diff(diff(x, n-1), 1)
        // For simplicity, we'll implement as multiple passes
        let current: ITensorHandle = input;
        for (let i = 0; i < n; i++) {
            const isLast = i === n - 1;
            current = diffKernelSingleOrder(
                current as WebGPUTensor<typeof dtype>,
                dim,
                isLast ? output : undefined
            );
        }
        return current;
    }

    return diffKernelSingleOrder(input as WebGPUTensor<typeof dtype>, dim, output);
}

/**
 * Single order diff kernel (n=1)
 */
function diffKernelSingleOrder<D extends DType>(
    input: WebGPUTensor<D>,
    dim: number,
    output?: WebGPUTensor<D>
): WebGPUTensor<D> {
    const device = WebGPUDeviceManager.device;
    const resolver = getGlobalDTypeResolver();

    const inputShape = [...input.shape] as number[];
    const inputStrides = [...input.strides] as number[];
    const rank = inputShape.length;
    const dtype = input.dtype;

    // Calculate output shape
    const outputShape = [...inputShape];
    outputShape[dim] = inputShape[dim] - 1;

    // Allocate if needed
    if (!output) {
        output = WebGPUTensor.createNew(outputShape as number[], dtype);
    }

    const outputStrides = [...output.strides] as number[];
    const outputNumel = outputShape.reduce((a, b) => a * b, 1);

    // Pipeline key
    const pipelineKey = `diff.${dtype}.r${rank}`;

    // BindGroupLayout
    const layoutKey = 'diff.layout';
    let bindGroupLayout = WebGPUPipelineManager.getBindGroupLayout(layoutKey);
    if (!bindGroupLayout) {
        bindGroupLayout = createBindGroupLayout(device);
        WebGPUPipelineManager.registerBindGroupLayout(layoutKey, bindGroupLayout);
    }

    // Pipeline
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);
    if (!pipeline) {
        const shaderCode = buildDiffShader({ dtype, rank });
        logger.debug(`Generated diff shader (dtype: ${dtype}, rank: ${rank})`);

        const module = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // Pad arrays to MAX_RANK
    const paddedInputShape = padToMax(inputShape);
    const paddedInputStrides = padToMax(inputStrides);
    const paddedOutputShape = padToMax(outputShape);
    const paddedOutputStrides = padToMax(outputStrides);

    // Get bytes per element
    const bytesPerElement = resolver.getDescriptor(dtype).gpuBytesPerElement;

    // Create Uniform Buffer
    const uniformBuffer = createUniformBuffer(device, {
        numel: outputNumel,
        n: 1,
        dim,
        rank,
        inputOffset: Math.floor(input.offset / bytesPerElement),
        outputOffset: Math.floor(output.offset / bytesPerElement),
        inputShape: paddedInputShape,
        inputStrides: paddedInputStrides,
        outputShape: paddedOutputShape,
        outputStrides: paddedOutputStrides,
    });

    // Create BindGroup
    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: input.buffer } },
            { binding: 2, resource: { buffer: output.buffer } },
        ],
    });

    // Dispatch
    const workgroupSize = 256;
    const workgroupCount = Math.ceil(outputNumel / workgroupSize);
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(workgroupCount);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    logger.debug(`Dispatched diff: shape=${inputShape}, dim=${dim}, output_shape=${outputShape}`);

    return output;
}

/**
 * Pad array to MAX_RANK elements
 */
function padToMax(arr: number[]): number[] {
    const result = [...arr];
    while (result.length < MAX_RANK) {
        result.push(arr.length > 0 ? 1 : 0);
    }
    return result.slice(0, MAX_RANK);
}

/**
 * Create BindGroupLayout for diff
 */
function createBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
    return device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'read-only-storage' },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'read-only-storage' },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            },
        ],
    });
}

/**
 * Create Uniform Buffer
 * 
 * Memory Layout (160 bytes):
 * [0-3]   numel
 * [4-7]   n
 * [8-11]  dim
 * [12-15] rank
 * [16-19] input_offset
 * [20-23] output_offset
 * [24-27] _pad0
 * [28-31] _pad1
 * [32-63] input_shape (8 * u32)
 * [64-95] input_strides (8 * i32)
 * [96-127] output_shape (8 * u32)
 * [128-159] output_strides (8 * i32)
 */
function createUniformBuffer(device: GPUDevice, uniforms: {
    numel: number;
    n: number;
    dim: number;
    rank: number;
    inputOffset: number;
    outputOffset: number;
    inputShape: number[];
    inputStrides: number[];
    outputShape: number[];
    outputStrides: number[];
}): GPUBuffer {
    const data = new ArrayBuffer(UNIFORM_BUFFER_SIZE);
    const view = new DataView(data);

    // Header (32 bytes)
    view.setUint32(0, uniforms.numel, true);
    view.setUint32(4, uniforms.n, true);
    view.setInt32(8, uniforms.dim, true);
    view.setUint32(12, uniforms.rank, true);
    view.setUint32(16, uniforms.inputOffset, true);
    view.setUint32(20, uniforms.outputOffset, true);
    view.setUint32(24, 0, true); // _pad0
    view.setUint32(28, 0, true); // _pad1

    // input_shape (32 bytes)
    for (let i = 0; i < MAX_RANK; i++) {
        view.setUint32(32 + i * 4, uniforms.inputShape[i] ?? 1, true);
    }

    // input_strides (32 bytes)
    for (let i = 0; i < MAX_RANK; i++) {
        view.setInt32(64 + i * 4, uniforms.inputStrides[i] ?? 0, true);
    }

    // output_shape (32 bytes)
    for (let i = 0; i < MAX_RANK; i++) {
        view.setUint32(96 + i * 4, uniforms.outputShape[i] ?? 1, true);
    }

    // output_strides (32 bytes)
    for (let i = 0; i < MAX_RANK; i++) {
        view.setInt32(128 + i * 4, uniforms.outputStrides[i] ?? 0, true);
    }

    // Create Storage Buffer (read-only in shader)
    // We use STORAGE usage to avoid strict alignment requirements (16 bytes for array elements)
    const buffer = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data));
    buffer.unmap();

    return buffer;
}
