/**
 * Normalize Kernel Executor
 * 
 * 执行 Softmax/LayerNorm/BatchNorm 等归一化操作
 * 
 * 使用 DirectContext 模式 (类似 Sort)
 * 
 * 工业级实现：原生支持 strided (非连续) 输入
 * 通过 shape/strides/offset 传递给 shader，无需预先克隆
 */

import type { ITensorHandle, DType } from '@kandle/types';
import type { NormalizeKernelParams, NormalizeShaderParams } from './types';
import { NORMALIZE_OPS } from './ops';
import { buildNormalizeShader } from './shaderBuilder';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUTensor } from '../../base/tensor';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { Logger } from '@kandle/utils';

const logger = new Logger('Normalize-Executor');

/**
 * 执行 Normalize 操作
 */
export function executeNormalize(
    dispatchKey: string,
    inputs: ITensorHandle[],
    params: NormalizeKernelParams
): ITensorHandle {
    const config = NORMALIZE_OPS[dispatchKey];
    if (!config) {
        throw new Error(`Unknown normalize operation: ${dispatchKey}`);
    }

    const device = WebGPUDeviceManager.device;
    const input = inputs[0];

    // 工业级实现：不再需要 ensureContiguous
    // shader 通过 strides/offset 原生支持 strided 访问

    const { shape, strides, offset, dtype } = input;
    const ndim = shape.length;

    // 解析参数 - 计算归一化维度
    let normalizedDims: number[];

    // layer_norm 和 rms_norm 使用 normalizedShape 参数
    if ((config.kind === 'layer_norm' || config.kind === 'rms_norm') && params.normalizedShape) {
        normalizedDims = computeNormalizedDims(shape, params.normalizedShape);
    } else {
        // 其他操作使用 dim 参数
        const dim = normalizeDim(params.dim, ndim, config.kind);
        normalizedDims = Array.isArray(dim) ? dim : [dim];
    }

    const reduceSize = normalizedDims.reduce((acc, d) => acc * shape[d], 1);
    const eps = params.eps ?? 1e-5;

    // 构建 shader 参数 - 包含完整的 strides/offset 信息
    const shaderParams: NormalizeShaderParams = {
        config,
        inputShape: shape,
        inputStrides: strides,
        inputOffset: offset,
        normalizedDims,
        reduceSize,
        hasWeight: !!params.weight,
        hasBias: !!params.bias,
        eps,
        p: params.p,
        numGroups: params.numGroups,
        dtype: dtype as string,
    };

    // Pipeline 缓存 key - include numGroups for group_norm
    const numGroupsPart = params.numGroups ? `.g${params.numGroups}` : '';
    const pipelineKey = `normalize.${dispatchKey}.${shape.join('x')}.${normalizedDims.join('_')}.${dtype}.w${params.weight ? 1 : 0}.b${params.bias ? 1 : 0}${numGroupsPart}`;

    // 尝试获取缓存的 pipeline
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    // Bind group layout key
    const layoutKey = `normalize.${dispatchKey}.w${params.weight ? 1 : 0}.b${params.bias ? 1 : 0}.rs${config.hasRunningStats ? 1 : 0}`;
    let bindGroupLayout = WebGPUPipelineManager.getBindGroupLayout(layoutKey);

    if (!bindGroupLayout) {
        bindGroupLayout = createNormalizeBindGroupLayout(device, config, !!params.weight, !!params.bias);
        WebGPUPipelineManager.registerBindGroupLayout(layoutKey, bindGroupLayout);
    }

    if (!pipeline) {
        // 生成 shader
        const shaderCode = buildNormalizeShader(shaderParams);

        // 创建 shader module
        const shaderModule = device.createShaderModule({ code: shaderCode });

        // Check for shader compilation errors (async but log)
        shaderModule.getCompilationInfo().then(info => {
            for (const message of info.messages) {
                if (message.type === 'error') {
                    console.error(`[Normalize Shader ERROR] Line ${message.lineNum}: ${message.message}`);
                }
            }
        });

        pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });

        // 缓存 pipeline
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // 创建输出 tensor
    const output = params.out as WebGPUTensor<typeof dtype> | undefined
        ?? WebGPUTensor.createNew([...shape] as number[], dtype as DType);

    // 构建 bind group entries
    const inputTensor = input as WebGPUTensor<typeof dtype>;
    const entries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: inputTensor.buffer } },
        { binding: 1, resource: { buffer: output.buffer } },
    ];

    let bindingIdx = 2;

    // BatchNorm 特殊处理: running_mean, running_var
    if (config.hasRunningStats && params.runningMean && params.runningVar) {
        entries.push({
            binding: bindingIdx++,
            resource: { buffer: (params.runningMean as WebGPUTensor<any>).buffer },
        });
        entries.push({
            binding: bindingIdx++,
            resource: { buffer: (params.runningVar as WebGPUTensor<any>).buffer },
        });
    }

    // Weight
    if (params.weight) {
        entries.push({
            binding: bindingIdx++,
            resource: { buffer: (params.weight as WebGPUTensor<any>).buffer },
        });
    }

    // Bias
    if (params.bias) {
        entries.push({
            binding: bindingIdx++,
            resource: { buffer: (params.bias as WebGPUTensor<any>).buffer },
        });
    }

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries,
    });

    // 计算 workgroup 数量
    const totalSize = shape.reduce((a, b) => a * b, 1);
    const batchSize = totalSize / reduceSize;
    const workgroupSize = 256;
    const numWorkgroups = Math.ceil(batchSize / workgroupSize);

    // 提交命令
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);

    return output;
}

/**
 * 创建 bind group layout
 */
function createNormalizeBindGroupLayout(
    device: GPUDevice,
    config: { hasRunningStats?: boolean },
    hasWeight: boolean,
    hasBias: boolean
): GPUBindGroupLayout {
    const entries: GPUBindGroupLayoutEntry[] = [
        // Input buffer
        {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'read-only-storage' },
        },
        // Output buffer
        {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'storage' },
        },
    ];

    let bindingIdx = 2;

    // BatchNorm: running_mean, running_var
    if (config.hasRunningStats) {
        entries.push({
            binding: bindingIdx++,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'read-only-storage' },
        });
        entries.push({
            binding: bindingIdx++,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'read-only-storage' },
        });
    }

    // Weight
    if (hasWeight) {
        entries.push({
            binding: bindingIdx++,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'read-only-storage' },
        });
    }

    // Bias
    if (hasBias) {
        entries.push({
            binding: bindingIdx++,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'read-only-storage' },
        });
    }

    return device.createBindGroupLayout({ entries });
}

/**
 * 规范化 dim 参数
 */
function normalizeDim(
    dim: number | number[] | undefined,
    ndim: number,
    kind: string
): number | number[] {
    // Softmax 默认 dim=-1
    if (dim === undefined) {
        switch (kind) {
            case 'softmax':
            case 'log_softmax':
            case 'softmin':
                return ndim - 1;
            case 'lp_normalize':
                return 1; // PyTorch F.normalize 默认 dim=1
            default:
                // layer_norm, rms_norm: 需要 normalizedShape 来确定
                return ndim - 1;
        }
    }

    // 处理负数索引
    if (typeof dim === 'number') {
        return dim < 0 ? dim + ndim : dim;
    }

    return dim.map(d => (d < 0 ? d + ndim : d));
}

/**
 * 根据 normalizedShape 计算 normalized dims
 */
export function computeNormalizedDims(
    inputShape: readonly number[],
    normalizedShape: readonly number[]
): number[] {
    const ndim = inputShape.length;
    const normDims = normalizedShape.length;

    // normalizedShape 对应输入的最后几个维度
    const dims: number[] = [];
    for (let i = 0; i < normDims; i++) {
        dims.push(ndim - normDims + i);
    }
    return dims;
}
