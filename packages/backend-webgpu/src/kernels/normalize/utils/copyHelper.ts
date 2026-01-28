/**
 * Normalize 模块的 Copy Helper
 * 
 * 提供轻量级的 strided -> contiguous 拷贝功能
 * 用于在 normalize 执行前确保输入连续
 */

import type { ITensorHandle, DType } from '@kandle/types';
import { WebGPUDeviceManager } from '../../../base/device';
import { WebGPUTensor } from '../../../base/tensor';
import { WebGPUPipelineManager } from '../../../pipelines/WebGPUPipelineManager';
import { createUniformBuffer } from '../../../base/uniformUtils';

/**
 * 执行 strided tensor 到 contiguous tensor 的拷贝
 * 
 * 这是一个专门为 normalize 模块设计的轻量级拷贝函数，
 * 避免了 TensorIterator 的复杂依赖。
 */
export function executeStridedToContiguousCopy(
    input: ITensorHandle
): ITensorHandle {
    const device = WebGPUDeviceManager.device;
    const { shape, strides, offset, dtype } = input;
    const rank = shape.length;
    const numel = shape.reduce((a, b) => a * b, 1);

    // 创建连续的输出 tensor
    const output = WebGPUTensor.createNew([...shape] as number[], dtype as DType);

    // Pipeline key 基于 rank 和 dtype
    const pipelineKey = `normalize.strided_copy.r${rank}.${dtype}`;

    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const shaderCode = buildStridedCopyShader(rank, dtype as string);
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // 创建 uniform buffer
    const uniformBuffer = createStridedCopyUniformBuffer(
        device, numel, rank, offset, shape, strides
    );

    const inputTensor = input as WebGPUTensor<typeof dtype>;

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: uniformBuffer } },
            { binding: 1, resource: { buffer: inputTensor.buffer } },
            { binding: 2, resource: { buffer: output.buffer } },
        ],
    });

    const workgroupSize = 256;
    const numWorkgroups = Math.ceil(numel / workgroupSize);

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
 * 生成 strided copy shader
 */
function buildStridedCopyShader(rank: number, dtype: string): string {
    // 根据 dtype 确定 WGSL 类型
    let wgslType: string;
    switch (dtype) {
        case 'float32':
            wgslType = 'f32';
            break;
        case 'int32':
            wgslType = 'i32';
            break;
        case 'uint32':
            wgslType = 'u32';
            break;
        default:
            // 其他类型默认用 f32 (WebGPU compute 不直接支持 f64, f16 等)
            wgslType = 'f32';
    }

    return `
// Strided to Contiguous Copy Shader
// Rank: ${rank}, dtype: ${dtype}

struct Uniforms {
    numel: u32,
    rank: u32,
    offset: u32,
    _pad: u32,
    // shape[0-3], shape[4-7] as vec4<u32>
    shape0: vec4<u32>,
    shape1: vec4<u32>,
    // strides[0-3], strides[4-7] as vec4<i32>
    strides0: vec4<i32>,
    strides1: vec4<i32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<${wgslType}>;
@group(0) @binding(2) var<storage, read_write> output: array<${wgslType}>;

fn get_shape(d: u32) -> u32 {
    if (d < 4u) {
        return uniforms.shape0[d];
    } else {
        return uniforms.shape1[d - 4u];
    }
}

fn get_stride(d: u32) -> i32 {
    if (d < 4u) {
        return uniforms.strides0[d];
    } else {
        return uniforms.strides1[d - 4u];
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.numel) {
        return;
    }
    
    // 从 flat index 计算多维坐标，然后用 strides 计算源索引
    var remaining = idx;
    var src_offset: i32 = i32(uniforms.offset);
    
    // 展开循环以避免动态 rank 访问问题
    ${generateUnrolledIndexLoop(rank)}
    
    output[idx] = input[u32(src_offset)];
}
`;
}

/**
 * 生成展开的索引计算循环
 */
function generateUnrolledIndexLoop(rank: number): string {
    if (rank === 0) {
        return '// Scalar case: no loop needed';
    }

    const lines: string[] = [];
    for (let d = 0; d < rank; d++) {
        // 计算该维度之后所有维度的乘积
        const suffixDims: string[] = [];
        for (let j = d + 1; j < rank; j++) {
            suffixDims.push(`get_shape(${j}u)`);
        }
        const dimSizeExpr = suffixDims.length > 0 ? suffixDims.join(' * ') : '1u';

        lines.push(`    {
        let dim_size_${d} = ${dimSizeExpr};
        let coord_${d} = remaining / dim_size_${d};
        remaining = remaining % dim_size_${d};
        src_offset += i32(coord_${d}) * get_stride(${d}u);
    }`);
    }
    return lines.join('\n');
}

/**
 * 创建 strided copy 的 uniform buffer
 */
function createStridedCopyUniformBuffer(
    device: GPUDevice,
    numel: number,
    rank: number,
    offset: number,
    shape: readonly number[],
    strides: readonly number[]
): GPUBuffer {
    // Layout aligned for vec4:
    // numel(4), rank(4), offset(4), _pad(4) = 16 bytes
    // shape0: vec4<u32> (16), shape1: vec4<u32> (16) = 32 bytes
    // strides0: vec4<i32> (16), strides1: vec4<i32> (16) = 32 bytes
    // Total: 80 bytes
    const uniformSize = 80;
    const data = new ArrayBuffer(uniformSize);
    const view = new DataView(data);
    let byteOffset = 0;

    // Header: numel, rank, offset, _pad
    view.setUint32(byteOffset, numel, true); byteOffset += 4;
    view.setUint32(byteOffset, rank, true); byteOffset += 4;
    view.setUint32(byteOffset, offset, true); byteOffset += 4;
    view.setUint32(byteOffset, 0, true); byteOffset += 4;  // padding

    // shape0: vec4<u32> (shape[0..3])
    for (let i = 0; i < 4; i++) {
        const val = i < rank ? shape[i] : 1;
        view.setUint32(byteOffset, val, true);
        byteOffset += 4;
    }

    // shape1: vec4<u32> (shape[4..7])
    for (let i = 4; i < 8; i++) {
        const val = i < rank ? shape[i] : 1;
        view.setUint32(byteOffset, val, true);
        byteOffset += 4;
    }

    // strides0: vec4<i32> (strides[0..3])
    for (let i = 0; i < 4; i++) {
        const val = i < rank ? strides[i] : 0;
        view.setInt32(byteOffset, val, true);
        byteOffset += 4;
    }

    // strides1: vec4<i32> (strides[4..7])
    for (let i = 4; i < 8; i++) {
        const val = i < rank ? strides[i] : 0;
        view.setInt32(byteOffset, val, true);
        byteOffset += 4;
    }

    return createUniformBuffer(data);
}
