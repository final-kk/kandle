/**
 * Arange Kernel Executor
 * 
 * 执行 arange 操作 - 在 GPU 上生成等差数列
 * 使用 TensorIterator 模式
 * 
 * @module kernels/factory/arangeExecutor
 */

import type { DType, ITensorIterator } from '@kandle/types';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { Logger } from '@kandle/utils';
import { buildArangeShader, isIntegerDtype } from './arangeShaderBuilder';
import { createUniformBuffer as createUniformBufferFromPool } from '../../base/uniformUtils';

const logger = new Logger('Arange-Executor');

/**
 * Uniform buffer size (16 bytes)
 * Layout:
 * - numel: u32
 * - outputOffset: u32
 * - start: f32/i32  (depends on dtype)
 * - step: f32/i32   (depends on dtype)
 */
const UNIFORM_BUFFER_SIZE = 16;

/**
 * Execute arange operation
 * 
 * Generates a 1-D tensor with values from [start, end) with step.
 * Formula: output[i] = start + i * step
 * 
 * @param iter - TensorIterator with output tensor configured and scalar args (start, end, step)
 */
export function executeArange(iter: ITensorIterator): void {
    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;

    // 1. Get scalar arguments
    const scalarArgs = (iter as any).getScalarArgs?.() as Record<string, number | boolean | string> | undefined;
    const start = (scalarArgs?.['start'] ?? 0) as number;
    const step = (scalarArgs?.['step'] ?? 1) as number;

    // 2. Get output info
    const numel = iter.outputNumel;
    const outputDtype = iter.output().dtype;
    const outputOffset = iter.output().offset;

    logger.debug(`Arange: start=${start}, step=${step}, numel=${numel}, dtype=${outputDtype}`);

    // Handle empty tensor case
    if (numel <= 0) {
        logger.debug('Arange: empty tensor, skipping GPU execution');
        return;
    }

    // 3. Pipeline key
    const pipelineKey = `factory.arange.${outputDtype}.wg${workgroupSize}`;

    // 4. Get or create pipeline
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const shaderCode = buildArangeShader(outputDtype, workgroupSize);
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
        logger.debug(`Created Arange pipeline: ${pipelineKey}`);
    }

    // 5. Create uniform buffer
    const uniformBuffer = createUniformBuffer(device, numel, outputOffset, start, step, outputDtype);

    // 6. Create bind group
    const outputBuffer = iter.output().tensorHandle.storage.buffer as GPUBuffer;
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: outputBuffer } },
            { binding: 1, resource: { buffer: uniformBuffer } },
        ],
    });

    // 7. Dispatch
    const numWorkgroups = Math.ceil(numel / workgroupSize);
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);

    logger.debug(`Arange complete: ${numel} elements, ${numWorkgroups} workgroups`);
}

/**
 * Create uniform buffer for arange operation
 * 
 * For integer dtypes, start and step are stored as i32.
 * For float dtypes, start and step are stored as f32.
 */
function createUniformBuffer(
    device: GPUDevice,
    numel: number,
    outputOffset: number,
    start: number,
    step: number,
    dtype: DType
): GPUBuffer {
    const data = new ArrayBuffer(UNIFORM_BUFFER_SIZE);
    const view = new DataView(data);

    // Common fields
    view.setUint32(0, numel, true);
    view.setUint32(4, outputOffset, true);

    // Start and step - choose format based on dtype
    const isInteger = isIntegerDtype(dtype);

    if (isInteger) {
        // Store as i32
        view.setInt32(8, Math.round(start), true);
        view.setInt32(12, Math.round(step), true);
    } else {
        // Store as f32
        view.setFloat32(8, start, true);
        view.setFloat32(12, step, true);
    }

    return createUniformBufferFromPool(data);
}
