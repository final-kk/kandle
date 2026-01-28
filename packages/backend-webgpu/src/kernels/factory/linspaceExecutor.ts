
import type { ITensorIterator } from '@kandle/types';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { createUniformBuffer as createUniformBufferFromPool } from '../../base/uniformUtils';
import { buildLinspaceShader } from './linspaceShaderBuilder';

const UNIFORM_BUFFER_SIZE = 16; // 4 * 4 bytes

export function executeLinspace(iter: ITensorIterator): void {
    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;

    // 1. Get scalar arguments
    const scalarArgs = (iter as any).getScalarArgs?.() as Record<string, number | boolean | string> | undefined;
    const start = (scalarArgs?.['start'] ?? 0) as number;
    const end = (scalarArgs?.['end'] ?? 0) as number;
    // steps is numel

    const numel = iter.outputNumel;
    const outputDtype = iter.output().dtype;
    const outputOffset = iter.output().offset;

    if (numel <= 0) return;

    const pipelineKey = `factory.linspace.${outputDtype}.wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const shaderCode = buildLinspaceShader(outputDtype, workgroupSize);
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // Uniform: numel (u32), offset (u32), start (f32), end (f32)
    const data = new ArrayBuffer(UNIFORM_BUFFER_SIZE);
    const view = new DataView(data);
    view.setUint32(0, numel, true);
    view.setUint32(4, outputOffset, true);
    view.setFloat32(8, start, true);
    view.setFloat32(12, end, true);

    const uniformBuffer = createUniformBufferFromPool(data);

    const outputBuffer = iter.output().tensorHandle.storage.buffer as GPUBuffer;
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: outputBuffer } },
            { binding: 1, resource: { buffer: uniformBuffer } },
        ],
    });

    const numWorkgroups = Math.ceil(numel / workgroupSize);
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
}
