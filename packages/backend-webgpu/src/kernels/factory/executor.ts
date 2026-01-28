/**
 * Factory Kernel Executor
 * 
 * 执行工厂操作（张量创建）
 * 使用 DirectContext 模式
 */

import type { DType, DirectContext } from '@kandle/types';
import { Logger } from '@kandle/utils';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { getGlobalDTypeResolver } from '../../base/DTypeResolver';
import { buildEyeShader } from './shaderBuilder';
import { createUniformBuffer } from '../../base/uniformUtils';

const logger = new Logger('Factory-Executor');

// ============================================================================
// Helper
// ============================================================================

function getElementByteSize(dtype: DType): number {
    const resolver = getGlobalDTypeResolver();
    const desc = resolver.getDescriptor(dtype);
    return desc.gpuBytesPerElement;
}

// ============================================================================
// Eye Executor
// ============================================================================

/**
 * 执行 Eye 操作
 * 
 * @param ctx DirectContext - 包含 output 和 scalars (n, m)
 */
export function executeEye(ctx: DirectContext): void {
    const output = ctx.outs![0];
    const n = (ctx.scalars['n'] ?? ctx.metadata?.['n'] ?? output.shape[0]) as number;
    const m = (ctx.scalars['m'] ?? ctx.metadata?.['m'] ?? output.shape[1] ?? n) as number;

    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;

    const numel = n * m;
    const dtype = output.dtype;

    logger.debug(`Eye: n=${n}, m=${m}, dtype=${dtype}`);

    // Pipeline key
    const pipelineKey = `factory_eye-${dtype}-wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const shaderCode = buildEyeShader(n, m, dtype, workgroupSize);
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
        logger.debug(`Created Eye pipeline: ${pipelineKey}`);
    }

    // 创建 uniform buffer
    // Layout: numel(4), n(4), m(4), outputOffset(4) = 16 bytes
    const data = new ArrayBuffer(16);
    const u32View = new Uint32Array(data);
    u32View[0] = numel;
    u32View[1] = n;
    u32View[2] = m;
    u32View[3] = output.offset / getElementByteSize(dtype);
    const uniformBuffer = createUniformBuffer(data);

    // 创建 bind group
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: output.storage.buffer as GPUBuffer } },
            { binding: 1, resource: { buffer: uniformBuffer } },
        ],
    });

    // 执行
    const numWorkgroups = Math.ceil(numel / workgroupSize);
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);

    logger.debug(`Eye complete: ${numel} elements, ${numWorkgroups} workgroups`);
}
