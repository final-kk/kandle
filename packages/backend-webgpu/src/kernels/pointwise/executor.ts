/**
 * v5 Pointwise Kernel Executor
 * 
 * Unified executor for all pointwise operations using POINTWISE_OPS config
 * Supports both contiguous (fast path) and strided (general path) access
 * 
 * IMPORTANT: Uses correct WGSL memory alignment for uniform buffers
 */

import { ITensorIterator } from '@kandle/types';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { getGlobalDTypeResolver } from '../../base/DTypeResolver';
import { Logger } from '@kandle/utils';
import { POINTWISE_OPS } from './ops';
import { buildPointwiseShader } from './shaderBuilder';
import type { PointwiseOpConfig } from './types';
import { UniformBufferPool } from '../../base/UniformBufferPool';

const logger = new Logger('Pointwise-Executor');
const MAX_RANK = 8;
const MAX_WORKGROUPS_PER_DIMENSION = 65535;

/**
 * 计算多维 workgroup 分布
 * 
 * WebGPU 限制每个维度最多 65535 个 workgroups
 * 对于大 tensor，需要将 workgroups 分布到多个维度
 * 
 * @param total 总 workgroup 数量
 * @returns {x, y, z} 三维 workgroup 分布
 */
function computeWorkgroupDimensions(total: number): { x: number; y: number; z: number } {
    if (total <= MAX_WORKGROUPS_PER_DIMENSION) {
        return { x: total, y: 1, z: 1 };
    }

    // 需要使用多维 dispatch
    const x = MAX_WORKGROUPS_PER_DIMENSION;
    const remaining = Math.ceil(total / x);

    if (remaining <= MAX_WORKGROUPS_PER_DIMENSION) {
        return { x, y: remaining, z: 1 };
    }

    // 极端情况：需要三维
    const y = MAX_WORKGROUPS_PER_DIMENSION;
    const z = Math.ceil(remaining / y);

    if (z > MAX_WORKGROUPS_PER_DIMENSION) {
        throw new Error(
            `Tensor too large: requires ${total} workgroups, ` +
            `exceeds maximum of ${MAX_WORKGROUPS_PER_DIMENSION ** 3}`
        );
    }

    return { x, y, z };
}

export function executePointwise(iter: ITensorIterator, dispatchKey: string): void {
    const opConfig = POINTWISE_OPS[dispatchKey];

    if (!opConfig) {
        throw new Error(`Unknown pointwise operation: ${dispatchKey}`);
    }

    const device = WebGPUDeviceManager.device;
    const resolver = getGlobalDTypeResolver();

    // 1. Process Scalar arguments (defaults + sentinels)
    const scalarArgs = processScalarArgs(iter, opConfig);

    // 2. Pipeline Key (includes all shader-affecting factors)
    const pipelineKey = computePipelineKey(iter, dispatchKey);

    // 3. BindGroupLayout
    const layoutKey = `pointwise.layout.${iter.numInputs}`;
    let bindGroupLayout = WebGPUPipelineManager.getBindGroupLayout(layoutKey);
    if (!bindGroupLayout) {
        bindGroupLayout = createBindGroupLayout(device, iter.numInputs);
        WebGPUPipelineManager.registerBindGroupLayout(layoutKey, bindGroupLayout);
    }

    // 4. Pipeline (cached)
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);
    if (!pipeline) {
        const shaderCode = buildPointwiseShader(iter, dispatchKey, opConfig, scalarArgs);
        logger.debug(`Generated shader for ${dispatchKey} (${iter.isContiguous ? 'fast' : 'general'} path)`);

        const module = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // 5. Uniform Buffer (includes shape, strides, offsets, scalarArgs)
    const uniformBuffer = iter.isContiguous
        ? createFastPathUniformBuffer(device, iter, scalarArgs)
        : createGeneralPathUniformBuffer(device, iter, scalarArgs);

    // 6. Handle Buffer Aliasing (In-place operations)
    const { finalBuffers, tempBuffers } = handleAliasing(iter, device);

    // 7. BindGroup
    const bindGroup = createBindGroup(device, bindGroupLayout, uniformBuffer, iter, finalBuffers);

    // 8. Dispatch
    // WebGPU 限制每个维度最多 65535 个 workgroups
    // 对于大 tensor，需要使用多维 dispatch
    const totalWorkgroups = Math.ceil(iter.outputNumel / 64);
    const { x, y, z } = computeWorkgroupDimensions(totalWorkgroups);

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(x, y, z);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);

    // 9. Destroy temporary buffers after GPU work completes
    // Note: Dawn caches destroyed buffers internally for reuse
    if (tempBuffers.length > 0) {
        device.queue.onSubmittedWorkDone().then(() => {
            tempBuffers.forEach(buf => buf.destroy());
        });
    }
}

/**
 * Process Scalar arguments: apply defaults and sentinels
 */
function processScalarArgs(
    iter: ITensorIterator,
    opConfig: PointwiseOpConfig
): Record<string, number> {
    const result: Record<string, number> = {};
    // Cast to access getScalarArgs() which exists on TensorIterator implementation but not in ITensorIterator interface
    const iterScalars = (iter as any).getScalarArgs() as Record<string, number | boolean | string>;

    // 1. Get values from iter
    for (const [key, value] of Object.entries(iterScalars)) {
        if (typeof value === 'number') {
            result[key] = value;
        } else if (typeof value === 'boolean') {
            result[key] = value ? 1 : 0;
        }
    }

    // 2. Apply defaults
    if (opConfig.scalarDefaults) {
        for (const [key, defaultValue] of Object.entries(opConfig.scalarDefaults)) {
            if (result[key] === undefined) {
                result[key] = defaultValue;
            }
        }
    }

    // 3. Apply sentinels (Optional parameters)
    if (opConfig.scalarSentinels) {
        for (const [key, sentinel] of Object.entries(opConfig.scalarSentinels)) {
            if (result[key] === undefined || iterScalars[key] === undefined) {
                result[key] = sentinel;
            }
        }
    }

    return result;
}

function computePipelineKey(iter: ITensorIterator, dispatchKey: string): string {
    const inputDtypes = Array.from({ length: iter.numInputs }, (_, i) =>
        iter.input(i).dtype
    ).join('-');
    const outputDtype = iter.output().dtype;
    const computeDtype = iter.computeDtype;
    const rank = iter.outputShape.length;
    const path = iter.isContiguous ? 'fast' : `general-r${rank}-i${iter.numInputs}`;

    return `pointwise.${dispatchKey}.${inputDtypes}.${outputDtype}.${computeDtype}.${path}`;
}

function createBindGroupLayout(device: GPUDevice, numInputs: number): GPUBindGroupLayout {
    const entries: GPUBindGroupLayoutEntry[] = [];

    // Uniforms @binding(0)
    entries.push({
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' }
    });

    // Input buffers @binding(1), @binding(2), ...
    for (let i = 0; i < numInputs; i++) {
        entries.push({
            binding: i + 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'read-only-storage' }
        });
    }

    // Output buffer @binding(numInputs + 1)
    entries.push({
        binding: numInputs + 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' }
    });

    return device.createBindGroupLayout({ entries });
}

/**
 * Create uniform buffer for FAST PATH (contiguous)
 * Layout: numel, offset_input0, offset_input1, ..., offset_output, scalar_args...
 */
function createFastPathUniformBuffer(
    device: GPUDevice,
    iter: ITensorIterator,
    scalarArgs: Record<string, number>
): GPUBuffer {
    const numScalars = Object.keys(scalarArgs).length;
    // numel(4) + offsets(4 * (numInputs + 1)) + scalars(4 * n), aligned to 16
    const baseSize = 4 + (iter.numInputs + 1) * 4 + numScalars * 4;
    const uniformSize = Math.ceil(baseSize / 16) * 16;

    const uniformBuffer = UniformBufferPool.getInstance().acquire(
        uniformSize,
        undefined,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );

    const uniformData = new ArrayBuffer(uniformSize);
    const uniformView = new DataView(uniformData);
    let offset = 0;

    // numel
    uniformView.setUint32(offset, iter.outputNumel, true);
    offset += 4;

    // Input offsets
    for (let i = 0; i < iter.numInputs; i++) {
        uniformView.setUint32(offset, iter.input(i).offset, true);
        offset += 4;
    }

    // Output offset
    uniformView.setUint32(offset, iter.output().offset, true);
    offset += 4;

    // Scalar args (SORTED)
    const sortedKeys = Object.keys(scalarArgs).sort();
    for (const key of sortedKeys) {
        uniformView.setFloat32(offset, scalarArgs[key], true);
        offset += 4;
    }

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);
    return uniformBuffer;
}

/**
 * Create uniform buffer for GENERAL PATH (strided)
 * 
 * Memory Layout (aligned to 16-byte boundaries with vec4):
 * - numel: u32 (4), rank: u32 (4), padding (8) = 16 bytes
 * - shape0: vec4<u32> (16), shape1: vec4<u32> (16) = 32 bytes
 * - For each input:
 *   - strides0: vec4<u32> (16), strides1: vec4<u32> (16) = 32 bytes
 *   - offset: u32 (4) + padding (12) = 16 bytes
 * - Output:
 *   - strides0: vec4<u32> (16), strides1: vec4<u32> (16) = 32 bytes
 *   - offset: u32 (4) + padding if scalars (12) = 16 bytes
 * - Scalars: f32 each (with padding to 16)
 */
function createGeneralPathUniformBuffer(
    device: GPUDevice,
    iter: ITensorIterator,
    scalarArgs: Record<string, number>
): GPUBuffer {
    const numScalars = Object.keys(scalarArgs).length;
    const rank = iter.outputShape.length;

    // Calculate size:
    // Header: 16 bytes (numel, rank, pad, pad)
    // Shape: 32 bytes (shape0: vec4, shape1: vec4)
    // Each input: 48 bytes (strides0: vec4, strides1: vec4, offset + pad: 16)
    // Output: 48 bytes (strides0: vec4, strides1: vec4, offset + pad: 16)
    // Scalars: 4 each + padding
    const baseSize = 16 + 32 + (iter.numInputs + 1) * 48 + (numScalars > 0 ? ((numScalars * 4) + 12) : 0);
    const uniformSize = Math.ceil(baseSize / 16) * 16;

    const uniformBuffer = UniformBufferPool.getInstance().acquire(
        uniformSize,
        undefined,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );

    const uniformData = new ArrayBuffer(uniformSize);
    const uniformView = new DataView(uniformData);
    let byteOffset = 0;

    // Header: numel, rank, _pad0, _pad1
    uniformView.setUint32(byteOffset, iter.outputNumel, true); byteOffset += 4;
    uniformView.setUint32(byteOffset, rank, true); byteOffset += 4;
    uniformView.setUint32(byteOffset, 0, true); byteOffset += 4; // _pad0
    uniformView.setUint32(byteOffset, 0, true); byteOffset += 4; // _pad1

    // Shape: shape0 (vec4), shape1 (vec4)
    for (let i = 0; i < 4; i++) {
        const dim = i < rank ? iter.outputShape[i] : 1;
        uniformView.setUint32(byteOffset, dim, true);
        byteOffset += 4;
    }
    for (let i = 4; i < 8; i++) {
        const dim = i < rank ? iter.outputShape[i] : 1;
        uniformView.setUint32(byteOffset, dim, true);
        byteOffset += 4;
    }

    // Inputs: strides0, strides1, offset + padding
    for (let inputIdx = 0; inputIdx < iter.numInputs; inputIdx++) {
        const input = iter.input(inputIdx);

        // strides0: vec4<u32>
        for (let i = 0; i < 4; i++) {
            const stride = i < input.strides.length ? input.strides[i] : 0;
            uniformView.setUint32(byteOffset, stride, true);
            byteOffset += 4;
        }

        // strides1: vec4<u32>
        for (let i = 4; i < 8; i++) {
            const stride = i < input.strides.length ? input.strides[i] : 0;
            uniformView.setUint32(byteOffset, stride, true);
            byteOffset += 4;
        }

        // offset + padding (16 bytes total)
        uniformView.setUint32(byteOffset, input.offset, true); byteOffset += 4;
        uniformView.setUint32(byteOffset, 0, true); byteOffset += 4; // _pad
        uniformView.setUint32(byteOffset, 0, true); byteOffset += 4; // _pad
        uniformView.setUint32(byteOffset, 0, true); byteOffset += 4; // _pad
    }

    // Output: strides0, strides1, offset + padding
    const output = iter.output();

    // strides0: vec4<u32>
    for (let i = 0; i < 4; i++) {
        const stride = i < output.strides.length ? output.strides[i] : 0;
        uniformView.setUint32(byteOffset, stride, true);
        byteOffset += 4;
    }

    // strides1: vec4<u32>
    for (let i = 4; i < 8; i++) {
        const stride = i < output.strides.length ? output.strides[i] : 0;
        uniformView.setUint32(byteOffset, stride, true);
        byteOffset += 4;
    }

    // offset
    uniformView.setUint32(byteOffset, output.offset, true); byteOffset += 4;

    // Scalar args (if any)
    const sortedKeys = Object.keys(scalarArgs).sort();
    if (sortedKeys.length > 0) {
        // Add padding for output offset
        uniformView.setUint32(byteOffset, 0, true); byteOffset += 4; // _pad_out
        uniformView.setUint32(byteOffset, 0, true); byteOffset += 4; // _pad_out2
        uniformView.setUint32(byteOffset, 0, true); byteOffset += 4; // _pad_out3

        for (const key of sortedKeys) {
            uniformView.setFloat32(byteOffset, scalarArgs[key], true);
            byteOffset += 4;
        }
    }

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);
    return uniformBuffer;
}

function handleAliasing(iter: ITensorIterator, device: GPUDevice): {
    finalBuffers: GPUBuffer[];
    tempBuffers: GPUBuffer[];
} {
    const finalBuffers: GPUBuffer[] = [];
    const tempBuffers: GPUBuffer[] = [];

    // Check for buffer aliasing (in-place ops)
    const outputBuffer = iter.output().tensorHandle.storage.buffer as GPUBuffer;

    for (let i = 0; i < iter.numInputs; i++) {
        const inputBuffer = iter.input(i).tensorHandle.storage.buffer as GPUBuffer;
        if (inputBuffer === outputBuffer) {
            // Create temporary buffer (Dawn caches destroyed buffers for reuse)
            const tempBuffer = device.createBuffer({
                size: inputBuffer.size,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
            });
            const commandEncoder = device.createCommandEncoder();
            commandEncoder.copyBufferToBuffer(inputBuffer, 0, tempBuffer, 0, inputBuffer.size);
            device.queue.submit([commandEncoder.finish()]);
            finalBuffers.push(tempBuffer);
            tempBuffers.push(tempBuffer);
        } else {
            finalBuffers.push(inputBuffer);
        }
    }

    return { finalBuffers, tempBuffers };
}

function createBindGroup(
    device: GPUDevice,
    layout: GPUBindGroupLayout,
    uniformBuffer: GPUBuffer,
    iter: ITensorIterator,
    inputBuffers: GPUBuffer[]
): GPUBindGroup {
    const entries: GPUBindGroupEntry[] = [];

    // Uniforms
    entries.push({
        binding: 0,
        resource: { buffer: uniformBuffer }
    });

    // Input buffers
    for (let i = 0; i < inputBuffers.length; i++) {
        entries.push({
            binding: i + 1,
            resource: { buffer: inputBuffers[i] }
        });
    }

    // Output buffer
    const outputBuffer = iter.output().tensorHandle.storage.buffer as GPUBuffer;
    entries.push({
        binding: inputBuffers.length + 1,
        resource: { buffer: outputBuffer }
    });

    return device.createBindGroup({ layout, entries });
}
