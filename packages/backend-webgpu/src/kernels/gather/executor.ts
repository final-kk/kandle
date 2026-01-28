/**
 * Gather Kernel Executor
 * 
 * 执行 index_select 等索引选择操作
 * 
 * 使用 DirectContext 模式
 * 
 * 工业级实现：原生支持 strided (非连续) 输入
 * 通过 shape/strides/offset 传递给 shader，无需预先克隆
 */

import type { ITensorHandle, DType } from '@kandle/types';
import type { IndexSelectParams, IndexSelectShaderParams } from './types';
import { buildIndexSelectShader } from './shaderBuilder';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUTensor } from '../../base/tensor';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { Logger } from '@kandle/utils';

const logger = new Logger('Gather-Executor');

/**
 * 执行 index_select 操作
 * 
 * 工业级实现：原生支持非连续输入
 * - 通过 strides/offset 在 shader 中计算物理地址
 * - 无需预先克隆，避免不必要的内存拷贝
 * 
 * @param self 源张量 (支持非连续)
 * @param index 索引张量 (1D, int32/int64, 支持非连续)
 * @param params 参数 { dim }
 * @param output 预分配的输出张量 (必须连续)
 */
export function executeIndexSelect(
    self: ITensorHandle,
    index: ITensorHandle,
    params: IndexSelectParams,
    output: ITensorHandle
): void {
    const device = WebGPUDeviceManager.device;

    // 工业级实现：不再需要 ensureContiguous
    // shader 通过 strides/offset 原生支持 strided 访问

    const { shape: inputShape, strides: inputStrides, offset: inputOffset, dtype } = self;
    const { dim } = params;
    const ndim = inputShape.length;

    // 验证
    if (index.shape.length !== 1) {
        throw new Error(`index_select: index must be 1D, got ${index.shape.length}D`);
    }
    if (dim < 0 || dim >= ndim) {
        throw new Error(`index_select: dim ${dim} out of range [0, ${ndim})`);
    }

    // 输出形状和步幅
    const outputShape = output.shape;
    const outputStrides = output.strides;
    const outputSize = outputShape.reduce((a, b) => a * b, 1);
    const indexLength = index.shape[0];

    // 获取 index 的 stride 和 offset (index 是 1D)
    const indexStride = index.strides[0];
    const indexOffset = index.offset;

    // 构建 shader 参数 - 包含完整的 strides/offset 信息
    const shaderParams: IndexSelectShaderParams = {
        inputShape,
        inputStrides,
        inputOffset,
        indexLength,
        indexStride,
        indexOffset,
        outputShape,
        outputStrides,
        dim,
        dtype: dtype as DType,
        outputSize,
    };

    // Pipeline 缓存 key - 包含 strides 特征以区分不同内存布局
    // 使用 strides 签名而非 `contiguous` 标志，支持任意布局
    const stridesKey = inputStrides.join('_');
    const pipelineKey = `gather.index_select.${inputShape.join('x')}.s${stridesKey}.d${dim}.idx${indexLength}.idxs${indexStride}.${dtype}`;

    // 尝试获取缓存的 pipeline
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    // Bind group layout key
    const layoutKey = 'gather.index_select';
    let bindGroupLayout = WebGPUPipelineManager.getBindGroupLayout(layoutKey);

    if (!bindGroupLayout) {
        bindGroupLayout = createIndexSelectBindGroupLayout(device);
        WebGPUPipelineManager.registerBindGroupLayout(layoutKey, bindGroupLayout);
    }

    if (!pipeline) {
        // 生成 shader
        const shaderCode = buildIndexSelectShader(shaderParams);

        // 创建 shader module
        const shaderModule = device.createShaderModule({ code: shaderCode });

        // Check for shader compilation errors (async but log)
        shaderModule.getCompilationInfo().then(info => {
            for (const message of info.messages) {
                if (message.type === 'error') {
                    console.error(`[IndexSelect Shader ERROR] Line ${message.lineNum}: ${message.message}`);
                    console.error(`Shader code:\n${shaderCode}`);
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

    // 构建 bind group
    const inputTensor = self as WebGPUTensor<typeof dtype>;
    const indexTensor = index as WebGPUTensor<typeof index.dtype>;
    const outputTensor = output as WebGPUTensor<typeof dtype>;

    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: inputTensor.buffer } },
            { binding: 1, resource: { buffer: indexTensor.buffer } },
            { binding: 2, resource: { buffer: outputTensor.buffer } },
        ],
    });

    // 计算 workgroup 数量
    const workgroupSize = 256;
    const numWorkgroups = Math.ceil(outputSize / workgroupSize);

    // 提交命令
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
}

/**
 * 创建 index_select bind group layout
 * 
 * bindings:
 * - 0: input (read-only)
 * - 1: index (read-only)
 * - 2: output (read-write)
 */
function createIndexSelectBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
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
                buffer: { type: 'storage' },
            },
        ],
    });
}
