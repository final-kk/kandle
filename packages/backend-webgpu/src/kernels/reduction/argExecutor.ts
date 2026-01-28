/**
 * ArgMax/ArgMin Executor
 * 
 * 专门的 argmax/argmin 归约执行器
 * 
 * argmax/argmin 与普通归约不同：
 * - 输出是索引 (int64 在 WebGPU 降级为 int32)
 * - 需要同时跟踪最值和对应的索引
 * - 支持全局和维度归约
 */

import { ITensorIterator } from '@kandle/types';
import { Logger } from '@kandle/utils';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import {
    buildArgDimReductionShader,
    buildArgGlobalReductionShader,
} from './argShaderBuilder';
import { createUniformBuffer } from '../../base/uniformUtils';

const logger = new Logger('Arg-Executor');

/**
 * argmax/argmin 执行入口
 */
export function executeArgReduction(iter: ITensorIterator, dispatchKey: 'argmax' | 'argmin'): void {
    if (!iter.isReduction) {
        throw new Error('Iterator must be a reduction operation');
    }

    // 判断是全局归约还是维度归约
    const isGlobalReduction = iter.outputShape.length === 0 ||
        (iter.outputShape.length === 1 && iter.outputShape[0] === 1 && iter.reductionShape.length > 0);

    logger.debug(`executeArgReduction: ${dispatchKey}, global=${isGlobalReduction}, outputShape=${iter.outputShape}, reductionShape=${iter.reductionShape}`);

    if (isGlobalReduction) {
        executeArgGlobalReduction(iter, dispatchKey);
    } else {
        executeArgDimReduction(iter, dispatchKey);
    }
}

/**
 * 全局 argmax/argmin
 * 扫描所有元素，返回全局最大/最小值的索引
 */
function executeArgGlobalReduction(iter: ITensorIterator, dispatchKey: 'argmax' | 'argmin'): void {
    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;
    const numel = iter.reductionNumel;
    const input = iter.input(0);
    const output = iter.output();

    logger.debug(`Global ${dispatchKey}: ${numel} elements`);

    // 检查输入是否连续
    const isContiguous = isInputContiguous(iter);
    const rank = iter.inputShape.length;

    // 生成 shader
    const shaderCode = buildArgGlobalReductionShader(iter, dispatchKey, workgroupSize, isContiguous, rank);
    const pipelineKey = isContiguous
        ? `arg_global.${dispatchKey}-${input.dtype}-wg${workgroupSize}`
        : `arg_global_strided.${dispatchKey}-${input.dtype}-r${rank}-wg${workgroupSize}`;

    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const module = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
        logger.debug(`Created pipeline: ${pipelineKey}`);
    }

    // 创建 uniform buffer
    let uniformBuffer: GPUBuffer;

    if (isContiguous) {
        const uniformData = new ArrayBuffer(16);
        const uniformView = new DataView(uniformData);
        uniformView.setUint32(0, numel, true);
        uniformView.setUint32(4, input.offset, true);
        uniformView.setUint32(8, output.offset, true);
        uniformBuffer = createUniformBuffer(uniformData);
    } else {
        uniformBuffer = createStridedArgUniformBuffer(device, iter, numel);
    }

    const inputBuffer = input.buffer as GPUBuffer;
    const outputBuffer = output.buffer as GPUBuffer;

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: inputBuffer } },
            { binding: 2, resource: { buffer: outputBuffer } },
        ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(1);  // 单 workgroup
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
}

/**
 * 维度 argmax/argmin
 * 沿指定维度归约，返回每个切片的最大/最小值索引
 */
function executeArgDimReduction(iter: ITensorIterator, dispatchKey: 'argmax' | 'argmin'): void {
    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;
    const input = iter.input(0);
    const output = iter.output();
    const outputNumel = iter.outputNumel;
    const reductionNumel = iter.reductionNumel;

    logger.debug(`Dim ${dispatchKey}: ${outputNumel} outputs, ${reductionNumel} reduction elements each`);

    // 生成 shader
    const shaderCode = buildArgDimReductionShader(iter, dispatchKey, workgroupSize);
    const pipelineKey = `arg_dim.${dispatchKey}-${input.dtype}-or${iter.outputShape.length}-rr${iter.reductionShape.length}-wg${workgroupSize}`;

    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const module = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
        logger.debug(`Created pipeline: ${pipelineKey}`);
    }

    // 创建 uniform buffer
    const uniformBuffer = createDimArgUniformBuffer(device, iter);

    const inputBuffer = input.buffer as GPUBuffer;
    const outputBuffer = output.buffer as GPUBuffer;

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: inputBuffer } },
            { binding: 2, resource: { buffer: outputBuffer } },
        ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(outputNumel);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
}

// ============================================================================
// Helper Functions
// ============================================================================

function isInputContiguous(iter: ITensorIterator): boolean {
    const input = iter.input(0);
    if (input.offset !== 0) return false;

    const shape = iter.inputShape;
    const strides = input.tensorHandle.strides;

    let expectedStride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
        if (shape[i] > 1 && strides[i] !== expectedStride) {
            return false;
        }
        expectedStride *= shape[i];
    }
    return true;
}

function createStridedArgUniformBuffer(device: GPUDevice, iter: ITensorIterator, numel: number): GPUBuffer {
    const input = iter.input(0);
    const output = iter.output();
    const shape = iter.inputShape;
    const strides = input.tensorHandle.strides;
    const rank = shape.length;

    // Layout: numel, rank, input_offset, output_offset (16), shape0/1 (32), strides0/1 (32) = 80 bytes
    const uniformSize = 80;
    const uniformData = new ArrayBuffer(uniformSize);
    const uniformView = new DataView(uniformData);
    let offset = 0;

    uniformView.setUint32(offset, numel, true); offset += 4;
    uniformView.setUint32(offset, rank, true); offset += 4;
    uniformView.setUint32(offset, input.offset, true); offset += 4;
    uniformView.setUint32(offset, output.offset, true); offset += 4;

    // shape0 (vec4)
    for (let i = 0; i < 4; i++) {
        const dim = i < rank ? shape[i] : 1;
        uniformView.setUint32(offset, dim, true); offset += 4;
    }
    // shape1 (vec4)
    for (let i = 4; i < 8; i++) {
        const dim = i < rank ? shape[i] : 1;
        uniformView.setUint32(offset, dim, true); offset += 4;
    }
    // strides0 (vec4)
    for (let i = 0; i < 4; i++) {
        const stride = i < rank ? strides[i] : 0;
        uniformView.setUint32(offset, stride, true); offset += 4;
    }
    // strides1 (vec4)
    for (let i = 4; i < 8; i++) {
        const stride = i < rank ? strides[i] : 0;
        uniformView.setUint32(offset, stride, true); offset += 4;
    }

    return createUniformBuffer(uniformData);
}

function createDimArgUniformBuffer(device: GPUDevice, iter: ITensorIterator): GPUBuffer {
    const input = iter.input(0);
    const output = iter.output();

    const outputRank = iter.outputShape.length;
    const parallelStrides = input.strides;
    const reductionStrides = input.reductionStrides || [];
    const reductionRank = iter.reductionShape.length;

    const reductionAxesSet = new Set(iter.reductionAxes);

    // 重建 input_parallel_strides 以匹配 output 维度
    let inputParallelStridesForOutput: number[];

    if (!iter.keepDims || outputRank === parallelStrides.length) {
        inputParallelStridesForOutput = [...parallelStrides];
    } else {
        inputParallelStridesForOutput = [];
        let parallelIdx = 0;
        for (let i = 0; i < outputRank; i++) {
            if (reductionAxesSet.has(i)) {
                inputParallelStridesForOutput.push(0);
            } else {
                inputParallelStridesForOutput.push(parallelStrides[parallelIdx] || 0);
                parallelIdx++;
            }
        }
    }

    // Layout same as standard dim reduction: 192 bytes
    const uniformSize = 192;
    const uniformData = new ArrayBuffer(uniformSize);
    const uniformView = new DataView(uniformData);
    let offset = 0;

    // Header
    uniformView.setUint32(offset, iter.outputNumel, true); offset += 4;
    uniformView.setUint32(offset, iter.reductionNumel, true); offset += 4;
    uniformView.setUint32(offset, iter.outputShape.length, true); offset += 4;
    uniformView.setUint32(offset, iter.reductionShape.length, true); offset += 4;

    // output_shape0/1
    for (let i = 0; i < 4; i++) {
        const dim = i < iter.outputShape.length ? iter.outputShape[i] : 1;
        uniformView.setUint32(offset, dim, true); offset += 4;
    }
    for (let i = 4; i < 8; i++) {
        const dim = i < iter.outputShape.length ? iter.outputShape[i] : 1;
        uniformView.setUint32(offset, dim, true); offset += 4;
    }

    // reduction_shape0/1
    for (let i = 0; i < 4; i++) {
        const dim = i < iter.reductionShape.length ? iter.reductionShape[i] : 1;
        uniformView.setUint32(offset, dim, true); offset += 4;
    }
    for (let i = 4; i < 8; i++) {
        const dim = i < iter.reductionShape.length ? iter.reductionShape[i] : 1;
        uniformView.setUint32(offset, dim, true); offset += 4;
    }

    // input_parallel_strides0/1
    for (let i = 0; i < 4; i++) {
        const stride = i < inputParallelStridesForOutput.length ? inputParallelStridesForOutput[i] : 0;
        uniformView.setUint32(offset, stride, true); offset += 4;
    }
    for (let i = 4; i < 8; i++) {
        const stride = i < inputParallelStridesForOutput.length ? inputParallelStridesForOutput[i] : 0;
        uniformView.setUint32(offset, stride, true); offset += 4;
    }

    // input_reduction_strides0/1
    for (let i = 0; i < 4; i++) {
        const stride = i < reductionStrides.length ? reductionStrides[i] : 0;
        uniformView.setUint32(offset, stride, true); offset += 4;
    }
    for (let i = 4; i < 8; i++) {
        const stride = i < reductionStrides.length ? reductionStrides[i] : 0;
        uniformView.setUint32(offset, stride, true); offset += 4;
    }

    // output_strides0/1
    const outputStrides = output.strides;
    for (let i = 0; i < 4; i++) {
        const stride = i < outputStrides.length ? outputStrides[i] : 0;
        uniformView.setUint32(offset, stride, true); offset += 4;
    }
    for (let i = 4; i < 8; i++) {
        const stride = i < outputStrides.length ? outputStrides[i] : 0;
        uniformView.setUint32(offset, stride, true); offset += 4;
    }

    // input_offset, output_offset
    uniformView.setUint32(offset, input.offset, true); offset += 4;
    uniformView.setUint32(offset, output.offset, true); offset += 4;

    return createUniformBuffer(uniformData);
}
