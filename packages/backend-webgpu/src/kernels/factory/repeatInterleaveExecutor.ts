/**
 * RepeatInterleave Kernel Executor
 * 
 * Executes repeat_interleave operation using WebGPU compute shaders.
 * 
 * Uses Direct Context pattern (like SortHandler):
 * - Receives (input, scalars, outs?) parameters directly
 * - Allocates output tensor
 * - Executes GPU kernel
 * 
 * repeat_interleave repeats each element of a tensor along a dimension.
 * [1, 2, 3] with repeats=2 -> [1, 1, 2, 2, 3, 3]
 */

import type { ITensorHandle, DType } from '@kandle/types';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { WebGPUTensor } from '../../base/tensor';
import { Logger } from '@kandle/utils';
import { buildRepeatInterleaveShader } from './repeatInterleaveShaderBuilder';
import type { RepeatInterleaveUniforms } from './repeatInterleaveTypes';
import { createUniformBuffer as createUniformBufferFromPool } from '../../base/uniformUtils';

const logger = new Logger('RepeatInterleave-Executor');

/**
 * Uniform buffer size (64 bytes)
 */
const UNIFORM_BUFFER_SIZE = 64;

/**
 * RepeatInterleave kernel implementation
 * 
 * Signature matches Sort/Sampling pattern:
 * (input: ITensorHandle, scalars: Record<string, unknown>, outs?: ITensorHandle[]) => ITensorHandle
 */
export function repeatInterleaveKernel(
    input: ITensorHandle,
    scalars: Record<string, unknown>,
    outs?: ITensorHandle[]
): ITensorHandle {
    const device = WebGPUDeviceManager.device;

    // 1. Get scalar args
    const repeats = (scalars['repeats'] ?? 1) as number;
    const dim = scalars['dim'] as number | null | undefined;
    const hasDim = dim !== null && dim !== undefined;

    // 2. Compute dimensions
    const inputShape = [...input.shape] as number[];
    const inputStrides = [...input.strides] as number[];
    const rank = inputShape.length;
    const dtype = input.dtype;

    // Normalize dim
    const normalizedDim = hasDim ? (dim < 0 ? dim + rank : dim) : -1;

    // 3. Calculate output shape
    let outputShape: number[];
    if (hasDim) {
        // Repeat along specific dimension
        outputShape = [...inputShape];
        outputShape[normalizedDim] *= repeats;
    } else {
        // Flatten first, then repeat
        const inputNumel = inputShape.reduce((a, b) => a * b, 1);
        outputShape = [inputNumel * repeats];
    }

    // 4. Allocate output tensor
    const output = outs?.[0] as WebGPUTensor<typeof dtype>
        ?? WebGPUTensor.createNew(outputShape as number[], dtype as DType);

    const outputNumel = outputShape.reduce((a, b) => a * b, 1);
    const inputNumel = inputShape.reduce((a, b) => a * b, 1);

    // 5. Pipeline key (depends on dtype, rank, hasDim)
    const pipelineKey = `repeat_interleave.${dtype}.r${rank}.${hasDim}`;

    // 6. BindGroupLayout
    const layoutKey = 'repeat_interleave.layout';
    let bindGroupLayout = WebGPUPipelineManager.getBindGroupLayout(layoutKey);
    if (!bindGroupLayout) {
        bindGroupLayout = createBindGroupLayout(device);
        WebGPUPipelineManager.registerBindGroupLayout(layoutKey, bindGroupLayout);
    }

    // 7. Pipeline
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);
    if (!pipeline) {
        const shaderCode = buildRepeatInterleaveShader({
            dtype: dtype as DType,
            rank,
            hasDim,
        });

        logger.debug(`Generated repeat_interleave shader (dtype: ${dtype}, rank: ${rank}, hasDim: ${hasDim})`);

        const module = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // 8. Pad shape and strides to 4 elements
    const paddedShape = padTo4(inputShape);
    const paddedStrides = padTo4(inputStrides);

    // 9. Get bytes per element for offset calculation
    const inputTensor = input as WebGPUTensor<typeof dtype>;
    const bytesPerElement = inputTensor.buffer.size / inputTensor.numel;

    // 10. Create Uniform Buffer
    const uniformBuffer = createUniformBuffer(device, {
        numel: outputNumel,
        repeats,
        inputNumel,
        rank,
        dim: normalizedDim,
        inputOffset: Math.floor(inputTensor.offset / bytesPerElement),
        outputOffset: Math.floor(output.offset / bytesPerElement),
        inputShape: paddedShape,
        inputStrides: paddedStrides,
    });

    // 11. Create BindGroup
    const inputBuffer = inputTensor.buffer;
    const outputBuffer = output.buffer;

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: inputBuffer } },
            { binding: 2, resource: { buffer: outputBuffer } },
        ],
    });

    // 12. Dispatch
    const workgroupSize = 256;
    const workgroupCount = Math.ceil(outputNumel / workgroupSize);
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(workgroupCount);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    logger.debug(`Dispatched repeat_interleave: shape=${inputShape}, repeats=${repeats}, dim=${normalizedDim}`);

    return output;
}

/**
 * Pad array to 4 elements
 */
function padTo4(arr: number[]): number[] {
    const result = [...arr];
    while (result.length < 4) {
        result.push(arr.length > 0 ? 1 : 0);
    }
    return result.slice(0, 4);
}

/**
 * Create BindGroupLayout for repeat_interleave
 */
function createBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
    return device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'uniform' },
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
 * Memory Layout (64 bytes):
 * [0-3]   numel
 * [4-7]   repeats
 * [8-11]  input_numel
 * [12-15] rank
 * [16-19] dim
 * [20-23] input_offset
 * [24-27] output_offset
 * [28-31] _pad
 * [32-47] input_shape (vec4<u32>)
 * [48-63] input_strides (vec4<i32>)
 */
function createUniformBuffer(device: GPUDevice, uniforms: RepeatInterleaveUniforms): GPUBuffer {
    const data = new ArrayBuffer(UNIFORM_BUFFER_SIZE);
    const view = new DataView(data);

    // Header (32 bytes)
    view.setUint32(0, uniforms.numel, true);
    view.setUint32(4, uniforms.repeats, true);
    view.setUint32(8, uniforms.inputNumel, true);
    view.setUint32(12, uniforms.rank, true);
    view.setInt32(16, uniforms.dim, true);
    view.setUint32(20, uniforms.inputOffset, true);
    view.setUint32(24, uniforms.outputOffset, true);
    view.setUint32(28, 0, true); // _pad

    // input_shape (16 bytes)
    for (let i = 0; i < 4; i++) {
        view.setUint32(32 + i * 4, uniforms.inputShape[i] ?? 1, true);
    }

    // input_strides (16 bytes)
    for (let i = 0; i < 4; i++) {
        view.setInt32(48 + i * 4, uniforms.inputStrides[i] ?? 0, true);
    }

    return createUniformBufferFromPool(data);
}
