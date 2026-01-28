/**
 * Scatter Kernel Executor
 *
 * 执行 scatter, scatter_add, scatter_reduce 操作
 *
 * 使用 DirectContext 模式，两阶段执行:
 * 1. 初始化阶段: copy self -> output 或初始化为单位元
 * 2. 散射阶段: 将 src 值写入/归约到对应位置
 * 
 * 工业级实现：原生支持 strided (非连续) 输入
 * 通过 shape/strides/offset 传递给 shader，无需预先克隆
 */

import type { ITensorHandle, DType } from '@kandle/types';
import type {
    ScatterParams,
    ScatterAddParams,
    ScatterReduceParams,
    ScatterShaderParams,
    ScatterOpConfig,
    ScatterReduceMode,
} from './types';
import { SCATTER_OPS, getScatterReduceConfig } from './ops';
import { buildScatterShader, buildScatterAddShader, buildScatterReduceShader } from './shaderBuilder';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUTensor } from '../../base/tensor';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { Logger } from '@kandle/utils';

const logger = new Logger('Scatter-Executor');

/**
 * 执行 scatter 操作
 */
export function executeScatter(
    self: ITensorHandle,
    index: ITensorHandle,
    src: ITensorHandle,
    params: ScatterParams,
    output: ITensorHandle
): void {
    const { dim } = params;
    const config: ScatterOpConfig = { ...SCATTER_OPS['scatter'] };

    executeScatterInternal(self, index, src, config, dim, output);
}

/**
 * 执行 scatter_add 操作
 */
export function executeScatterAdd(
    self: ITensorHandle,
    index: ITensorHandle,
    src: ITensorHandle,
    params: ScatterAddParams,
    output: ITensorHandle
): void {
    const { dim } = params;
    const config: ScatterOpConfig = { ...SCATTER_OPS['scatter_add'] };

    executeScatterAtomicInternal(self, index, src, config, dim, output, 'scatter_add');
}

/**
 * 执行 scatter_reduce 操作
 */
export function executeScatterReduce(
    self: ITensorHandle,
    index: ITensorHandle,
    src: ITensorHandle,
    params: ScatterReduceParams,
    output: ITensorHandle
): void {
    const { dim, reduce, includeSelf } = params;
    const config = getScatterReduceConfig(reduce, includeSelf);

    executeScatterAtomicInternal(self, index, src, config, dim, output, 'scatter_reduce');
}

/**
 * 内部执行函数: scatter (非原子)
 * 
 * 工业级实现：原生支持非连续输入
 */
function executeScatterInternal(
    self: ITensorHandle,
    index: ITensorHandle,
    src: ITensorHandle,
    config: ScatterOpConfig,
    dim: number,
    output: ITensorHandle
): void {
    const device = WebGPUDeviceManager.device;

    // 工业级实现：不再需要 ensureContiguous
    // shader 通过 strides/offset 原生支持 strided 访问

    const dtype = self.dtype as DType;
    const ndim = self.shape.length;

    // 计算尺寸
    const indexSize = index.shape.reduce((a, b) => a * b, 1);
    const outputSize = output.shape.reduce((a, b) => a * b, 1);

    // 构建 shader 参数 - 包含完整的 strides/offset 信息
    const shaderParams: ScatterShaderParams = {
        config,
        selfShape: self.shape,
        selfStrides: self.strides,
        selfOffset: self.offset,
        indexShape: index.shape,
        indexStrides: index.strides,
        indexOffset: index.offset,
        srcShape: src.shape,
        srcStrides: src.strides,
        srcOffset: src.offset,
        outputShape: output.shape,
        outputStrides: output.strides,
        dim,
        dtype,
        indexSize,
        outputSize,
    };

    // Pipeline 缓存 key - 包含 strides 特征以区分不同内存布局
    const pipelineKey = `scatter.scatter.s${self.shape.join('x')}_${self.strides.join('_')}.i${index.shape.join('x')}_${index.strides.join('_')}.src${src.shape.join('x')}_${src.strides.join('_')}.o${output.shape.join('x')}_${output.strides.join('_')}.d${dim}.${dtype}`;

    // 获取或创建 bind group layout
    const layoutKey = 'scatter.4buffer';
    let bindGroupLayout = WebGPUPipelineManager.getBindGroupLayout(layoutKey);

    if (!bindGroupLayout) {
        bindGroupLayout = create4BufferBindGroupLayout(device);
        WebGPUPipelineManager.registerBindGroupLayout(layoutKey, bindGroupLayout);
    }

    // 获取或创建 pipeline
    let copyPipeline = WebGPUPipelineManager.getPipeline(pipelineKey + '.copy');
    let scatterPipeline = WebGPUPipelineManager.getPipeline(pipelineKey + '.scatter');

    if (!copyPipeline || !scatterPipeline) {
        const shaderCode = buildScatterShader(shaderParams);
        const shaderModule = device.createShaderModule({ code: shaderCode });

        // 检查编译错误
        shaderModule.getCompilationInfo().then(info => {
            for (const message of info.messages) {
                if (message.type === 'error') {
                    console.error(`[Scatter Shader ERROR] Line ${message.lineNum}: ${message.message}`);
                    console.error(`Shader code:\n${shaderCode}`);
                }
            }
        });

        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

        copyPipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'copy_phase' },
        });

        scatterPipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'scatter_phase' },
        });

        WebGPUPipelineManager.registerPipeline(pipelineKey + '.copy', copyPipeline);
        WebGPUPipelineManager.registerPipeline(pipelineKey + '.scatter', scatterPipeline);
    }

    // 构建 bind group
    const selfTensor = self as WebGPUTensor<typeof dtype>;
    const indexTensor = index as WebGPUTensor<typeof index.dtype>;
    const srcTensor = src as WebGPUTensor<typeof src.dtype>;
    const outputTensor = output as WebGPUTensor<typeof dtype>;

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: selfTensor.buffer } },
            { binding: 1, resource: { buffer: indexTensor.buffer } },
            { binding: 2, resource: { buffer: srcTensor.buffer } },
            { binding: 3, resource: { buffer: outputTensor.buffer } },
        ],
    });

    // 计算 workgroup 数量
    const workgroupSize = 256;
    const copyWorkgroups = Math.ceil(outputSize / workgroupSize);
    const scatterWorkgroups = Math.ceil(indexSize / workgroupSize);

    // 提交命令
    const commandEncoder = device.createCommandEncoder();

    // Phase 1: Copy
    const copyPass = commandEncoder.beginComputePass();
    copyPass.setPipeline(copyPipeline);
    copyPass.setBindGroup(0, bindGroup);
    copyPass.dispatchWorkgroups(copyWorkgroups);
    copyPass.end();

    // Phase 2: Scatter
    const scatterPass = commandEncoder.beginComputePass();
    scatterPass.setPipeline(scatterPipeline);
    scatterPass.setBindGroup(0, bindGroup);
    scatterPass.dispatchWorkgroups(scatterWorkgroups);
    scatterPass.end();

    device.queue.submit([commandEncoder.finish()]);
}

/**
 * 内部执行函数: scatter_add / scatter_reduce (原子操作)
 * 
 * 工业级实现：原生支持非连续输入
 */
function executeScatterAtomicInternal(
    self: ITensorHandle,
    index: ITensorHandle,
    src: ITensorHandle,
    config: ScatterOpConfig,
    dim: number,
    output: ITensorHandle,
    opType: 'scatter_add' | 'scatter_reduce'
): void {
    const device = WebGPUDeviceManager.device;

    // 工业级实现：不再需要 ensureContiguous
    // shader 通过 strides/offset 原生支持 strided 访问

    const dtype = self.dtype as DType;
    const ndim = self.shape.length;

    // 计算尺寸
    const indexSize = index.shape.reduce((a, b) => a * b, 1);
    const outputSize = output.shape.reduce((a, b) => a * b, 1);

    // 构建 shader 参数 - 包含完整的 strides/offset 信息
    const shaderParams: ScatterShaderParams = {
        config,
        selfShape: self.shape,
        selfStrides: self.strides,
        selfOffset: self.offset,
        indexShape: index.shape,
        indexStrides: index.strides,
        indexOffset: index.offset,
        srcShape: src.shape,
        srcStrides: src.strides,
        srcOffset: src.offset,
        outputShape: output.shape,
        outputStrides: output.strides,
        dim,
        dtype,
        indexSize,
        outputSize,
    };

    // Pipeline 缓存 key - 包含 strides 特征以区分不同内存布局
    const reduceMode = config.reduceMode ?? 'sum';
    const includeSelf = config.includeSelf ?? true;
    const shapeKey = `s${self.shape.join('x')}_${self.strides.join('_')}.i${index.shape.join('x')}_${index.strides.join('_')}.src${src.shape.join('x')}_${src.strides.join('_')}.o${output.shape.join('x')}_${output.strides.join('_')}`;
    const pipelineKey = opType === 'scatter_add'
        ? `scatter.scatter_add.${shapeKey}.d${dim}.${dtype}`
        : `scatter.scatter_reduce.${shapeKey}.d${dim}.${dtype}.${reduceMode}.${includeSelf}`;

    // 获取或创建 bind group layout
    const layoutKey = 'scatter.4buffer';
    let bindGroupLayout = WebGPUPipelineManager.getBindGroupLayout(layoutKey);

    if (!bindGroupLayout) {
        bindGroupLayout = create4BufferBindGroupLayout(device);
        WebGPUPipelineManager.registerBindGroupLayout(layoutKey, bindGroupLayout);
    }

    // 获取或创建 pipeline
    const initEntryPoint = opType === 'scatter_add' ? 'copy_phase' : 'init_phase';
    const scatterEntryPoint = opType === 'scatter_add' ? 'scatter_add_phase' : 'scatter_reduce_phase';

    let initPipeline = WebGPUPipelineManager.getPipeline(pipelineKey + '.init');
    let scatterPipeline = WebGPUPipelineManager.getPipeline(pipelineKey + '.scatter');

    if (!initPipeline || !scatterPipeline) {
        const shaderCode = opType === 'scatter_add'
            ? buildScatterAddShader(shaderParams)
            : buildScatterReduceShader(shaderParams);

        const shaderModule = device.createShaderModule({ code: shaderCode });

        // 检查编译错误
        shaderModule.getCompilationInfo().then(info => {
            for (const message of info.messages) {
                if (message.type === 'error') {
                    console.error(`[${opType} Shader ERROR] Line ${message.lineNum}: ${message.message}`);
                    console.error(`Shader code:\n${shaderCode}`);
                }
            }
        });

        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

        initPipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: initEntryPoint },
        });

        scatterPipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: scatterEntryPoint },
        });

        WebGPUPipelineManager.registerPipeline(pipelineKey + '.init', initPipeline);
        WebGPUPipelineManager.registerPipeline(pipelineKey + '.scatter', scatterPipeline);
    }

    // 构建 bind group
    const selfTensor = self as WebGPUTensor<typeof dtype>;
    const indexTensor = index as WebGPUTensor<typeof index.dtype>;
    const srcTensor = src as WebGPUTensor<typeof src.dtype>;
    const outputTensor = output as WebGPUTensor<typeof dtype>;

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: selfTensor.buffer } },
            { binding: 1, resource: { buffer: indexTensor.buffer } },
            { binding: 2, resource: { buffer: srcTensor.buffer } },
            { binding: 3, resource: { buffer: outputTensor.buffer } },
        ],
    });

    // 计算 workgroup 数量
    const workgroupSize = 256;
    const initWorkgroups = Math.ceil(outputSize / workgroupSize);
    const scatterWorkgroups = Math.ceil(indexSize / workgroupSize);

    // 提交命令
    const commandEncoder = device.createCommandEncoder();

    // Phase 1: Initialize
    const initPass = commandEncoder.beginComputePass();
    initPass.setPipeline(initPipeline);
    initPass.setBindGroup(0, bindGroup);
    initPass.dispatchWorkgroups(initWorkgroups);
    initPass.end();

    // Phase 2: Scatter
    const scatterPass = commandEncoder.beginComputePass();
    scatterPass.setPipeline(scatterPipeline);
    scatterPass.setBindGroup(0, bindGroup);
    scatterPass.dispatchWorkgroups(scatterWorkgroups);
    scatterPass.end();

    device.queue.submit([commandEncoder.finish()]);
}

/**
 * 创建 4 个 buffer 的 bind group layout
 *
 * bindings:
 * - 0: self (read-only)
 * - 1: index (read-only)
 * - 2: src (read-only)
 * - 3: output (read-write)
 */
function create4BufferBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
    return device.createBindGroupLayout({
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'read-only-storage' },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'read-only-storage' },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'read-only-storage' },
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: 'storage' },
            },
        ],
    });
}
