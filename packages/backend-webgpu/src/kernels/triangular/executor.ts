/**
 * Triangular Kernel Executor (工业级实现)
 * 
 * 执行 triu/tril 三角矩阵操作
 * 使用 DirectContext 模式，不依赖 TensorIterator
 * 
 * 工业级特性：
 * - 原生支持 strided (非连续) 输入
 * - 无需自动克隆，直接在 shader 中处理 strided 访问
 * 
 * 参考: PyTorch ATen/native/TensorShape.cpp
 */

import type { DType, DirectContext } from '@kandle/types';
import { Logger } from '@kandle/utils';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { getGlobalDTypeResolver } from '../../base/DTypeResolver';
import { getComputeType } from '../../base/dtype';
import { TRIANGULAR_OPS } from './ops';
import { buildTriangularShader } from './shaderBuilder';
import type { TriangularShaderParams } from './types';

const logger = new Logger('Triangular-Executor');

// ============================================================================
// Helper
// ============================================================================

function getElementByteSize(dtype: DType): number {
    const resolver = getGlobalDTypeResolver();
    const desc = resolver.getDescriptor(dtype);
    return desc.gpuBytesPerElement;
}

// ============================================================================
// Main Entry Point
// ============================================================================

/**
 * 执行 Triangular 操作 (triu/tril)
 * 
 * 工业级实现：原生支持 strided 输入
 * 
 * @param ctx DirectContext - 包含 inputs, output, scalars
 * @param dispatchKey 操作名称 (triu/tril)
 */
export function executeTriangular(ctx: DirectContext, dispatchKey: string): void {
    const config = TRIANGULAR_OPS[dispatchKey];
    if (!config) {
        throw new Error(`Unknown Triangular operation: ${dispatchKey}`);
    }

    const input = ctx.inputs[0];
    const output = ctx.outs![0];
    const diagonal = (ctx.scalars['diagonal'] ?? ctx.metadata?.['diagonal'] ?? 0) as number;

    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;

    const shape = input.shape;
    const strides = input.strides;
    const inputOffset = input.offset / getElementByteSize(input.dtype); // 转为 element offset
    const numel = shape.reduce((a, b) => a * b, 1);
    const rank = shape.length;

    // 最后两维是 (M, N)
    const M = shape[rank - 2] || 1;
    const N = shape[rank - 1] || 1;

    logger.debug(`Triangular: ${dispatchKey}, shape=[${shape.join(', ')}], strides=[${strides.join(', ')}], offset=${inputOffset}, diagonal=${diagonal}`);

    // Pipeline key - 包含 strides 的 hash 以正确区分不同内存布局
    const stridesKey = strides.join('_');
    const computeDtype = input.dtype;
    const pipelineKey = `triangular.${dispatchKey}-${computeDtype}-wg${workgroupSize}-shape${shape.join('_')}-strides${stridesKey}-offset${inputOffset}-diag${diagonal}`;

    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const params: TriangularShaderParams = {
            config,
            inputShape: shape,
            inputStrides: strides,
            inputOffset,
            diagonal,
            wgslType: getComputeType(computeDtype),
            workgroupSize,
        };
        const shaderCode = buildTriangularShader(params);

        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
        logger.debug(`Created Triangular pipeline: ${pipelineKey}`);
    }

    // 创建 bind group
    // 现在不需要 uniform buffer，所有参数都通过 shader 常量传递
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: input.storage.buffer as GPUBuffer } },
            { binding: 1, resource: { buffer: output.storage.buffer as GPUBuffer } },
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

    logger.debug(`Triangular complete: ${numel} elements, ${numWorkgroups} workgroups`);
}
