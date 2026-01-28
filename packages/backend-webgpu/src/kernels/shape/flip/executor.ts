/**
 * Flip Kernel Executor
 *
 * Executes flip operation (reversing elements along specified dimensions).
 */

import type { ITensorHandle, DType } from '@kandle/types';
import { WebGPUDeviceManager } from '../../../base/device';
import { WebGPUPipelineManager } from '../../../pipelines/WebGPUPipelineManager';
import { WebGPUTensor } from '../../../base/tensor';
import { Logger } from '@kandle/utils';
import { buildFlipShader } from './shaderBuilder';
import { getGlobalDTypeResolver } from '../../../base/DTypeResolver';

const logger = new Logger('Flip-Executor');

const MAX_RANK = 8;

/**
 * Uniform buffer size
 * Header: 8 * 4 = 32 bytes
 * input_shape: 8 * 4 = 32 bytes
 * input_strides: 8 * 4 = 32 bytes
 * output_strides: 8 * 4 = 32 bytes
 * Total: 128 bytes
 */
const UNIFORM_BUFFER_SIZE = 128;

/**
 * Flip kernel implementation
 *
 * Signature: (inputs, scalars, outs?) => ITensorHandle
 */
export function flipKernel(
    inputs: ITensorHandle[],
    scalars: Record<string, unknown>,
    outs?: ITensorHandle[]
): ITensorHandle {
    const input = inputs[0];
    const device = WebGPUDeviceManager.device;
    const resolver = getGlobalDTypeResolver();

    // 1. Get dims parameter
    let dims = scalars['dims'] as number[];
    if (!Array.isArray(dims)) {
        throw new Error('flip: dims must be an array of integers');
    }

    // 2. Compute dimensions
    const inputShape = [...input.shape] as number[];
    const inputStrides = [...input.strides] as number[];
    const rank = inputShape.length;
    const dtype = input.dtype;

    // 3. Normalize dims and create flip mask
    let flipMask = 0;
    const normalizedDims: number[] = [];
    for (const dim of dims) {
        let d = dim < 0 ? dim + rank : dim;
        if (d < 0 || d >= rank) {
            throw new Error(`flip: dim ${dim} out of range for tensor of rank ${rank}`);
        }
        if (!normalizedDims.includes(d)) {
            normalizedDims.push(d);
            flipMask |= (1 << d);
        }
    }

    // 4. Output has same shape as input
    const outputShape = [...inputShape];
    const numel = outputShape.reduce((a, b) => a * b, 1);

    // 5. Allocate output tensor
    const output = outs?.[0] as WebGPUTensor<typeof dtype>
        ?? WebGPUTensor.createNew(outputShape as number[], dtype as DType);

    const outputStrides = [...output.strides] as number[];

    // 6. Pipeline key
    const pipelineKey = `flip.${dtype}.r${rank}`;

    // 7. BindGroupLayout
    const layoutKey = 'flip.layout';
    let bindGroupLayout = WebGPUPipelineManager.getBindGroupLayout(layoutKey);
    if (!bindGroupLayout) {
        bindGroupLayout = createBindGroupLayout(device);
        WebGPUPipelineManager.registerBindGroupLayout(layoutKey, bindGroupLayout);
    }

    // 8. Pipeline
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);
    if (!pipeline) {
        const shaderCode = buildFlipShader({ dtype, rank });
        logger.debug(`Generated flip shader (dtype: ${dtype}, rank: ${rank})`);

        const module = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // 9. Pad arrays to MAX_RANK
    const paddedInputShape = padToMax(inputShape);
    const paddedInputStrides = padToMax(inputStrides);
    const paddedOutputStrides = padToMax(outputStrides);

    // Get bytes per element
    const bytesPerElement = resolver.getDescriptor(dtype).gpuBytesPerElement;

    // 10. Create Uniform Buffer
    const uniformBuffer = createUniformBuffer(device, {
        numel,
        rank,
        inputOffset: Math.floor(input.offset / bytesPerElement),
        outputOffset: Math.floor(output.offset / bytesPerElement),
        flipMask,
        inputShape: paddedInputShape,
        inputStrides: paddedInputStrides,
        outputStrides: paddedOutputStrides,
    });

    // 11. Create BindGroup
    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: (input as WebGPUTensor<typeof dtype>).buffer } },
            { binding: 2, resource: { buffer: output.buffer } },
        ],
    });

    // 12. Dispatch
    const workgroupSize = 256;
    const workgroupCount = Math.ceil(numel / workgroupSize);
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(workgroupCount);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    logger.debug(`Dispatched flip: shape=${inputShape}, dims=[${normalizedDims}], flip_mask=0b${flipMask.toString(2)}`);

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
 * Create BindGroupLayout for flip
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
 * Memory Layout (128 bytes):
 * [0-3]   numel
 * [4-7]   rank
 * [8-11]  input_offset
 * [12-15] output_offset
 * [16-19] flip_mask (bitmask of dims to flip)
 * [20-23] _pad0
 * [24-27] _pad1
 * [28-31] _pad2
 * [32-63] input_shape (8 * u32)
 * [64-95] input_strides (8 * i32)
 * [96-127] output_strides (8 * i32)
 */
function createUniformBuffer(device: GPUDevice, uniforms: {
    numel: number;
    rank: number;
    inputOffset: number;
    outputOffset: number;
    flipMask: number;
    inputShape: number[];
    inputStrides: number[];
    outputStrides: number[];
}): GPUBuffer {
    const data = new ArrayBuffer(UNIFORM_BUFFER_SIZE);
    const view = new DataView(data);

    // Header (32 bytes)
    view.setUint32(0, uniforms.numel, true);
    view.setUint32(4, uniforms.rank, true);
    view.setUint32(8, uniforms.inputOffset, true);
    view.setUint32(12, uniforms.outputOffset, true);
    view.setUint32(16, uniforms.flipMask, true);
    view.setUint32(20, 0, true); // _pad0
    view.setUint32(24, 0, true); // _pad1
    view.setUint32(28, 0, true); // _pad2

    // input_shape (32 bytes)
    for (let i = 0; i < MAX_RANK; i++) {
        view.setUint32(32 + i * 4, uniforms.inputShape[i] ?? 1, true);
    }

    // input_strides (32 bytes)
    for (let i = 0; i < MAX_RANK; i++) {
        view.setInt32(64 + i * 4, uniforms.inputStrides[i] ?? 0, true);
    }

    // output_strides (32 bytes)
    for (let i = 0; i < MAX_RANK; i++) {
        view.setInt32(96 + i * 4, uniforms.outputStrides[i] ?? 0, true);
    }

    // Create Storage Buffer
    const buffer = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data));
    buffer.unmap();

    return buffer;
}
