/**
 * Random Kernel Executor
 * 
 * Executes random number generation kernels (rand, randn, randint)
 * using Philox 4x32-10 PRNG on WebGPU.
 */

import { ITensorIterator, RandomState } from '@kandle/types';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { Logger } from '@kandle/utils';
import { buildRandomShader } from './shaderBuilder';
import type { RandomOpType, RandomUniforms } from './types';
import { createUniformBuffer as createUniformBufferFromPool } from '../../base/uniformUtils';

const logger = new Logger('Random-Executor');

/**
 * Uniform buffer size (48 bytes, 3 × 16-byte aligned)
 */
const UNIFORM_BUFFER_SIZE = 48;

/**
 * Execute a random operation
 * 
 * @param iter - TensorIterator with output tensor configured
 * @param opType - Type of random operation ('rand' | 'randn' | 'randint')
 */
export function executeRandom(iter: ITensorIterator, opType: RandomOpType): void {
    const device = WebGPUDeviceManager.device;
    const state = RandomState.getInstance();

    // 1. Get Philox key from global state
    const [key0, key1] = state.getKey();

    // 2. Calculate output size and Philox call count
    const numel = iter.outputNumel;
    // Each Philox call generates 4 u32 values
    // For randn, we need 2 uniforms per normal, so 2 normals per Philox call
    const philoxCalls = opType === 'randn'
        ? Math.ceil(numel / 2)  // 2 normal values per Philox call
        : Math.ceil(numel / 4); // 4 values per Philox call

    // 3. Consume offset from global state
    const baseOffset = state.consumeOffset(philoxCalls);

    // 4. Get randint parameters if applicable
    const scalarArgs = (iter as any).getScalarArgs?.() as Record<string, number | boolean | string> | undefined;
    const low = (scalarArgs?.['low'] ?? 0) as number;
    const high = (scalarArgs?.['high'] ?? 1) as number;

    // 5. Get output dtype
    const outputDtype = iter.output().dtype;

    // 6. Pipeline key
    const pipelineKey = `random.${opType}.${outputDtype}`;

    // 7. BindGroupLayout
    const layoutKey = `random.layout`;
    let bindGroupLayout = WebGPUPipelineManager.getBindGroupLayout(layoutKey);
    if (!bindGroupLayout) {
        bindGroupLayout = createBindGroupLayout(device);
        WebGPUPipelineManager.registerBindGroupLayout(layoutKey, bindGroupLayout);
    }

    // 8. Pipeline (cached)
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);
    if (!pipeline) {
        const shaderCode = buildRandomShader({
            opType,
            outputDtype,
        });

        logger.debug(`Generated shader for ${opType} (dtype: ${outputDtype})`);

        const module = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // 9. Create Uniform Buffer
    const uniformBuffer = createUniformBuffer(device, {
        numel,
        outputOffset: iter.output().offset,
        key0,
        key1,
        baseOffset,
        low,
        high,
    });

    // 10. Create BindGroup
    const outputBuffer = iter.output().tensorHandle.storage.buffer as GPUBuffer;
    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: outputBuffer } },
        ],
    });

    // 11. Dispatch
    const workgroupCount = Math.ceil(numel / 64);
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(workgroupCount);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    logger.debug(`Dispatched ${opType} with ${numel} elements, ${workgroupCount} workgroups`);
}

/**
 * Create BindGroupLayout for random operations
 * 
 * Layout:
 * - @binding(0): uniform buffer (RandomUniforms)
 * - @binding(1): output storage buffer
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
                buffer: { type: 'storage' },
            },
        ],
    });
}

/**
 * Create Uniform Buffer for random operations
 * 
 * Memory Layout (48 bytes):
 * - [0-3]   numel: u32
 * - [4-7]   output_offset: u32
 * - [8-11]  _pad0: u32
 * - [12-15] _pad1: u32
 * - [16-19] key0: u32
 * - [20-23] key1: u32
 * - [24-27] base_offset: u32
 * - [28-31] _pad2: u32
 * - [32-35] low: i32
 * - [36-39] high: i32
 * - [40-43] _pad3: u32
 * - [44-47] _pad4: u32
 */
function createUniformBuffer(device: GPUDevice, uniforms: RandomUniforms): GPUBuffer {
    const data = new ArrayBuffer(UNIFORM_BUFFER_SIZE);
    const view = new DataView(data);

    // Basic info (16 bytes)
    view.setUint32(0, uniforms.numel, true);
    view.setUint32(4, uniforms.outputOffset, true);
    view.setUint32(8, 0, true);  // _pad0
    view.setUint32(12, 0, true); // _pad1

    // Philox Key (16 bytes)
    view.setUint32(16, uniforms.key0, true);
    view.setUint32(20, uniforms.key1, true);
    view.setUint32(24, uniforms.baseOffset, true);
    view.setUint32(28, 0, true); // _pad2

    // randint params (16 bytes)
    view.setInt32(32, uniforms.low, true);
    view.setInt32(36, uniforms.high, true);
    view.setUint32(40, 0, true); // _pad3
    view.setUint32(44, 0, true); // _pad4

    return createUniformBufferFromPool(data);
}
