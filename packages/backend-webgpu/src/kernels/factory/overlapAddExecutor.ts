/**
 * Overlap-Add Executor
 * 
 * Implements the overlap-add algorithm for iSTFT reconstruction.
 * 
 * Input: (n_frames, n_fft) or (batch, n_frames, n_fft)
 * Output: (output_len,) or (batch, output_len)
 * 
 * Each output sample accumulates contributions from overlapping frames.
 */

import { ITensorHandle } from '@kandle/types';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { getGlobalDTypeResolver } from '../../base/DTypeResolver';
import { createUniformBuffer } from '../../base/uniformUtils';
import { Logger } from '@kandle/utils';

const logger = new Logger('OverlapAdd-Executor');

export interface OverlapAddConfig {
    n_frames: number;
    n_fft: number;
    hop_length: number;
    output_len: number;
    batch_size: number;
    dtype: string;
}

function buildOverlapAddShader(config: OverlapAddConfig): string {
    const { dtype } = config;
    const resolver = getGlobalDTypeResolver();
    const storageType = resolver.getDescriptor(dtype as any).wgslStorageType;

    // Each thread handles one output sample
    return `
struct Uniforms {
    n_frames: u32,
    n_fft: u32,
    hop_length: u32,
    output_len: u32,
    batch_size: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: array<${storageType}>;
@group(0) @binding(2) var<storage, read_write> output: array<${storageType}>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_output = uniforms.batch_size * uniforms.output_len;
    let idx = global_id.x;
    
    if (idx >= total_output) {
        return;
    }
    
    // Decompose index into batch and output position
    let batch = idx / uniforms.output_len;
    let out_pos = idx % uniforms.output_len;
    
    // Accumulate contributions from all frames that overlap this position
    var sum: ${storageType} = ${storageType}(0.0);
    
    // Find first frame that could contribute to this position
    // Frame t contributes to positions [t * hop_length, t * hop_length + n_fft)
    // We want: t * hop_length <= out_pos < t * hop_length + n_fft
    // => out_pos - n_fft < t * hop_length <= out_pos
    // => (out_pos - n_fft + 1) / hop_length <= t <= out_pos / hop_length
    
    let first_frame_approx = select(0u, (out_pos + 1u - uniforms.n_fft) / uniforms.hop_length, out_pos >= uniforms.n_fft - 1u);
    
    for (var t: u32 = first_frame_approx; t < uniforms.n_frames; t++) {
        let frame_start = t * uniforms.hop_length;
        let frame_end = frame_start + uniforms.n_fft;
        
        if (frame_start > out_pos) {
            break; // No more frames contribute
        }
        
        if (out_pos < frame_end) {
            // This frame contributes
            let in_frame_pos = out_pos - frame_start;
            let input_idx = batch * uniforms.n_frames * uniforms.n_fft + t * uniforms.n_fft + in_frame_pos;
            sum = sum + input[input_idx];
        }
    }
    
    output[idx] = sum;
}
`;
}

import { DirectContext } from '@kandle/types';

export function executeOverlapAdd(ctx: DirectContext): void {
    const { inputs, scalars, outs } = ctx;
    const input = inputs[0];
    const output = outs![0];
    const hop_length = scalars['hop_length'] as number;

    const device = WebGPUDeviceManager.device;

    const inputShape = input.shape;
    const ndim = inputShape.length;
    const isBatched = ndim === 3;

    const batch_size = isBatched ? inputShape[0] : 1;
    const n_frames = isBatched ? inputShape[1] : inputShape[0];
    const n_fft = isBatched ? inputShape[2] : inputShape[1];
    const output_len = output.shape[output.shape.length - 1];

    const config: OverlapAddConfig = {
        n_frames,
        n_fft,
        hop_length,
        output_len,
        batch_size,
        dtype: input.dtype,
    };

    // Create pipeline
    const pipelineKey = `overlap_add_${n_frames}_${n_fft}_${hop_length}_${batch_size}_${input.dtype}`;

    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);
    if (!pipeline) {
        const shaderCode = buildOverlapAddShader(config);
        logger.debug(`Creating overlap-add pipeline: ${pipelineKey}`);

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

    // Create uniform buffer
    const uniformData = new ArrayBuffer(32);
    const uniformView = new DataView(uniformData);
    uniformView.setUint32(0, n_frames, true);
    uniformView.setUint32(4, n_fft, true);
    uniformView.setUint32(8, hop_length, true);
    uniformView.setUint32(12, output_len, true);
    uniformView.setUint32(16, batch_size, true);

    const uniformBuffer = createUniformBuffer(uniformData);

    // Create bind group
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

    // Dispatch
    const totalOutput = batch_size * output_len;
    const workgroupCount = Math.ceil(totalOutput / 256);

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();

    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(workgroupCount);

    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
}
