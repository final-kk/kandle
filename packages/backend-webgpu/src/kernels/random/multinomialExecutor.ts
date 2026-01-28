/**
 * Multinomial Kernel Executor
 * 
 * Executes multinomial sampling using WebGPU compute shaders.
 * 
 * Uses Direct Context pattern (like SortHandler, not TensorIterator):
 * - Receives (input, scalars, outs?) parameters directly
 * - Allocates output tensor
 * - Executes GPU kernel
 * 
 * Algorithm:
 * 1. Read input probabilities (weights)
 * 2. For each sample, generate uniform random number
 * 3. Find the index where cumulative sum passes the threshold
 * 
 * Notes:
 * - Currently only supports replacement=true (with replacement)
 * - Input can be 1D [classes] or 2D [batch, classes]
 * - Output is [batch, numSamples] or [numSamples]
 */

import type { ITensorHandle } from '@kandle/types';
import { RandomState } from '@kandle/types';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { WebGPUTensor } from '../../base/tensor';
import { Logger } from '@kandle/utils';
import { buildMultinomialShader } from './multinomialShaderBuilder';
import type { MultinomialUniforms } from './multinomialTypes';
import { createUniformBuffer as createUniformBufferFromPool } from '../../base/uniformUtils';

const logger = new Logger('Multinomial-Executor');

/**
 * Uniform buffer size (48 bytes, 3 × 16-byte aligned)
 */
const UNIFORM_BUFFER_SIZE = 48;

/**
 * Multinomial kernel implementation
 * 
 * Signature matches SortKernelImpl pattern:
 * (input: ITensorHandle, scalars: Record<string, unknown>, outs?: ITensorHandle[]) => ITensorHandle
 */
export function multinomialKernel(
    input: ITensorHandle,
    scalars: Record<string, unknown>,
    outs?: ITensorHandle[]
): ITensorHandle {
    const device = WebGPUDeviceManager.device;
    const state = RandomState.getInstance();

    // 1. Parse shape: [batch, classes] or [classes]
    const inputShape = input.shape;
    const is1D = inputShape.length === 1;
    const batchSize = is1D ? 1 : inputShape[0];
    const numClasses = is1D ? inputShape[0] : inputShape[1];

    // 2. Get sample parameters from scalars
    const numSamples = (scalars['numSamples'] ?? 1) as number;
    const replacement = (scalars['replacement'] ?? true) as boolean;

    if (!replacement) {
        // TODO: Implement without-replacement sampling
        logger.warn('Multinomial without replacement not yet implemented, using with replacement');
    }

    // 3. Compute output shape
    const outputShape = is1D ? [numSamples] : [batchSize, numSamples];

    // 4. Allocate output tensor (int32 for indices)
    const output = outs?.[0] as WebGPUTensor<'int32'>
        ?? WebGPUTensor.createNew(outputShape as number[], 'int32');

    // 5. Get Philox key from global state
    const [key0, key1] = state.getKey();

    // 6. Calculate Philox call count and consume offset
    const totalSamples = batchSize * numSamples;
    const philoxCalls = Math.ceil(totalSamples / 4);
    const baseOffset = state.consumeOffset(philoxCalls);

    // 7. Pipeline key
    const pipelineKey = `multinomial.${numClasses}.${replacement}`;

    // 8. BindGroupLayout
    const layoutKey = 'multinomial.layout';
    let bindGroupLayout = WebGPUPipelineManager.getBindGroupLayout(layoutKey);
    if (!bindGroupLayout) {
        bindGroupLayout = createBindGroupLayout(device);
        WebGPUPipelineManager.registerBindGroupLayout(layoutKey, bindGroupLayout);
    }

    // 9. Pipeline (cached per numClasses and replacement mode)
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);
    if (!pipeline) {
        const shaderCode = buildMultinomialShader({
            numClasses,
            replacement: true, // Always true for now
        });

        logger.debug(`Generated shader for multinomial (numClasses: ${numClasses})`);

        const module = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // 10. Create Uniform Buffer
    const inputTensor = input as WebGPUTensor<'float32'>;
    const uniformBuffer = createUniformBuffer(device, {
        batchSize,
        numClasses,
        numSamples,
        replacement: replacement ? 1 : 0,
        inputOffset: inputTensor.offset / 4,  // float32 elements
        outputOffset: output.offset / 4,       // int32 elements
        key0,
        key1,
        baseOffset,
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
    const workgroupCount = Math.ceil(totalSamples / 64);
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(workgroupCount);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    logger.debug(`Dispatched multinomial: batch=${batchSize}, classes=${numClasses}, samples=${numSamples}, workgroups=${workgroupCount}`);

    return output;
}

/**
 * Create BindGroupLayout for multinomial
 * 
 * Layout:
 * - @binding(0): uniform buffer (MultinomialUniforms)
 * - @binding(1): input storage buffer (probabilities, read-only)
 * - @binding(2): output storage buffer (indices, read-write)
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
 * Create Uniform Buffer for multinomial
 * 
 * Memory Layout (48 bytes):
 * - [0-3]   batch_size: u32
 * - [4-7]   num_classes: u32
 * - [8-11]  num_samples: u32
 * - [12-15] replacement: u32
 * - [16-19] input_offset: u32
 * - [20-23] output_offset: u32
 * - [24-27] _pad0: u32
 * - [28-31] _pad1: u32
 * - [32-35] key0: u32
 * - [36-39] key1: u32
 * - [40-43] base_offset: u32
 * - [44-47] _pad2: u32
 */
function createUniformBuffer(device: GPUDevice, uniforms: MultinomialUniforms): GPUBuffer {
    const data = new ArrayBuffer(UNIFORM_BUFFER_SIZE);
    const view = new DataView(data);

    // Basic info (16 bytes)
    view.setUint32(0, uniforms.batchSize, true);
    view.setUint32(4, uniforms.numClasses, true);
    view.setUint32(8, uniforms.numSamples, true);
    view.setUint32(12, uniforms.replacement, true);

    // Offsets and padding (16 bytes)
    view.setUint32(16, uniforms.inputOffset, true);
    view.setUint32(20, uniforms.outputOffset, true);
    view.setUint32(24, 0, true);  // _pad0
    view.setUint32(28, 0, true);  // _pad1

    // Philox Key (16 bytes)
    view.setUint32(32, uniforms.key0, true);
    view.setUint32(36, uniforms.key1, true);
    view.setUint32(40, uniforms.baseOffset, true);
    view.setUint32(44, 0, true);  // _pad2

    return createUniformBufferFromPool(data);
}
