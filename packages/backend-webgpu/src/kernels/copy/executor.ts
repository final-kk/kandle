/**
 * Copy Kernel Executor
 * 
 * Executes copy operations:
 * - cast: type conversion
 * - contiguous: strided to contiguous conversion
 * - clone: direct copy
 * 
 * v5 FIX: Uses Uniforms for shape/strides, enabling pipeline caching
 * Uses correct WGSL memory alignment with vec4 types
 */

import { DType, ITensorIterator } from '@kandle/types';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { isContiguousStrides, Logger } from '@kandle/utils';
import type { CopyOpConfig, CopyVariant } from './types';
import { buildCopyShader, getCopyPipelineKey } from './shaderBuilder';
import { createUniformBuffer as createUniformBufferFromPool } from '../../base/uniformUtils';

const logger = new Logger('Copy-Executor');
const MAX_RANK = 8;

/**
 * Execute copy operation using TensorIterator
 */
export function executeCopy(iter: ITensorIterator, variant: CopyVariant): void {
    const device = WebGPUDeviceManager.device;

    const input = iter.input(0);
    const output = iter.output();

    // Compute numel from shape
    const numel = input.shape.reduce((a, b) => a * b, 1);

    // Build configuration
    const config: CopyOpConfig = {
        inputDtype: input.dtype,
        outputDtype: output.dtype,
        shape: input.shape,
        inputStrides: input.strides as number[],
        outputStrides: output.strides as number[],
        inputOffset: input.offset,
        outputOffset: output.offset,
        numel
    };

    // v5: Use cacheable pipeline key (only depends on rank, dtypes, path)
    const pipelineKey = getCopyPipelineKey(config);

    // Get or create pipeline
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);
    if (!pipeline) {
        // Generate shader
        const shaderCode = buildCopyShader(config);
        logger.debug(`Creating copy pipeline for key: ${pipelineKey}`);

        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main'
            }
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // Create uniform buffer with runtime shape/strides
    const uniformBuffer = createUniformBuffer(device, config);

    // Create bind group
    const inputBuffer = input.buffer as GPUBuffer;
    const outputBuffer = output.buffer as GPUBuffer;

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: inputBuffer } },
            { binding: 2, resource: { buffer: outputBuffer } }
        ]
    });

    // Dispatch
    const workgroupCount = Math.ceil(numel / 256);
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();

    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(workgroupCount);

    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
}

/**
 * Create uniform buffer for copy operation
 * Fast path: numel, offset_input, offset_output
 * Strided path: numel, rank, offset_input, offset_output, shape0/1 (vec4), input_strides0/1 (vec4), output_strides0/1 (vec4)
 */
function createUniformBuffer(device: GPUDevice, config: CopyOpConfig): GPUBuffer {
    const { shape, inputStrides, outputStrides, inputOffset, outputOffset, numel } = config;
    const rank = shape.length;

    const isInputContiguous = isContiguousStrides(shape, inputStrides);
    const isOutputContiguous = isContiguousStrides(shape, outputStrides);
    const isFastPath = isInputContiguous && isOutputContiguous && inputOffset === 0 && outputOffset === 0;

    if (isFastPath) {
        return createFastPathUniformBuffer(device, numel, inputOffset, outputOffset);
    } else {
        return createStridedPathUniformBuffer(device, config);
    }
}

function createFastPathUniformBuffer(
    device: GPUDevice,
    numel: number,
    inputOffset: number,
    outputOffset: number
): GPUBuffer {
    // Layout: numel(4), offset_input(4), offset_output(4) = 12, aligned to 16
    const uniformSize = 16;
    const uniformData = new ArrayBuffer(uniformSize);
    const uniformView = new DataView(uniformData);

    uniformView.setUint32(0, numel, true);
    uniformView.setUint32(4, inputOffset, true);
    uniformView.setUint32(8, outputOffset, true);

    return createUniformBufferFromPool(uniformData);
}

function createStridedPathUniformBuffer(device: GPUDevice, config: CopyOpConfig): GPUBuffer {
    const { shape, inputStrides, outputStrides, inputOffset, outputOffset, numel } = config;
    const rank = shape.length;

    // Layout with vec4 alignment:
    // numel(4), rank(4), offset_input(4), offset_output(4) = 16 bytes
    // shape0: vec4<u32> (16), shape1: vec4<u32> (16) = 32 bytes  
    // input_strides0: vec4<i32> (16), input_strides1: vec4<i32> (16) = 32 bytes
    // output_strides0: vec4<i32> (16), output_strides1: vec4<i32> (16) = 32 bytes
    // Total: 112 bytes, aligned to 128
    const uniformSize = 128;
    const uniformData = new ArrayBuffer(uniformSize);
    const uniformView = new DataView(uniformData);
    let offset = 0;

    // Header: numel, rank, offset_input, offset_output (16 bytes)
    uniformView.setUint32(offset, numel, true); offset += 4;
    uniformView.setUint32(offset, rank, true); offset += 4;
    uniformView.setUint32(offset, inputOffset, true); offset += 4;
    uniformView.setUint32(offset, outputOffset, true); offset += 4;

    // shape0: vec4<u32> (shape[0..3])
    for (let i = 0; i < 4; i++) {
        const dim = i < rank ? shape[i] : 1;
        uniformView.setUint32(offset, dim, true);
        offset += 4;
    }

    // shape1: vec4<u32> (shape[4..7])
    for (let i = 4; i < 8; i++) {
        const dim = i < rank ? shape[i] : 1;
        uniformView.setUint32(offset, dim, true);
        offset += 4;
    }

    // input_strides0: vec4<i32> (input_strides[0..3])
    for (let i = 0; i < 4; i++) {
        const stride = i < rank ? inputStrides[i] : 0;
        uniformView.setInt32(offset, stride, true);
        offset += 4;
    }

    // input_strides1: vec4<i32> (input_strides[4..7])
    for (let i = 4; i < 8; i++) {
        const stride = i < rank ? inputStrides[i] : 0;
        uniformView.setInt32(offset, stride, true);
        offset += 4;
    }

    // output_strides0: vec4<i32> (output_strides[0..3])
    for (let i = 0; i < 4; i++) {
        const stride = i < rank ? outputStrides[i] : 0;
        uniformView.setInt32(offset, stride, true);
        offset += 4;
    }

    // output_strides1: vec4<i32> (output_strides[4..7])
    for (let i = 4; i < 8; i++) {
        const stride = i < rank ? outputStrides[i] : 0;
        uniformView.setInt32(offset, stride, true);
        offset += 4;
    }

    return createUniformBufferFromPool(uniformData);
}
