/**
 * WindowFunc Kernel Executor
 * 
 * Implements window function generation using WebGPU compute shaders.
 */

import type { WindowFuncKernelArgs, WindowConfig } from './types';
import { WebGPUDeviceManager } from '../../base/device';
import { WebGPUPipelineManager } from '../../pipelines/WebGPUPipelineManager';
import { createUniformBuffer } from '../../base/uniformUtils';
import { getGlobalDTypeResolver } from '../../base/DTypeResolver';
import { WGSL_CONSTANTS } from '../../base/dtype';
import type { DType } from '@kandle/types';
import { computeBesselI0 } from '@kandle/utils';

/**
 * Execute window function generation kernel
 */
export function executeWindowFunc(args: WindowFuncKernelArgs): void {
    const { output, windowLength, denominator, config } = args;

    const device = WebGPUDeviceManager.device;
    const workgroupSize = WebGPUDeviceManager.optimalWorkgroupSize;
    const dtype = output.dtype;

    // Select shader builder based on template
    let shaderCode: string;
    let uniformData: ArrayBuffer;

    switch (config.template) {
        case 'generalized_cosine':
            shaderCode = buildGeneralizedCosineShader(dtype, workgroupSize);
            uniformData = createGeneralizedCosineUniforms(windowLength, denominator, config.coeffs, output);
            break;
        case 'linear':
            shaderCode = buildLinearShader(dtype, workgroupSize, config.windowType);
            uniformData = createLinearUniforms(windowLength, denominator, output);
            break;
        case 'kaiser':
            shaderCode = buildKaiserShader(dtype, workgroupSize);
            uniformData = createKaiserUniforms(windowLength, denominator, config.beta, output);
            break;
        case 'gaussian':
            shaderCode = buildGaussianShader(dtype, workgroupSize);
            uniformData = createGaussianUniforms(windowLength, denominator, config.std, output);
            break;
        case 'exponential':
            shaderCode = buildExponentialShader(dtype, workgroupSize);
            uniformData = createExponentialUniforms(windowLength, denominator, config.tau, config.center, output);
            break;
        default:
            throw new Error(`Unsupported window template: ${(config as any).template}`);
    }

    // Get or create pipeline
    // For linear template, include windowType in key since bartlett and triang have different shaders
    const pipelineKey = config.template === 'linear'
        ? `windowfunc_linear_${config.windowType}-${dtype}-wg${workgroupSize}`
        : `windowfunc_${config.template}-${dtype}-wg${workgroupSize}`;
    let pipeline = WebGPUPipelineManager.getPipeline(pipelineKey);

    if (!pipeline) {
        const shaderModule = device.createShaderModule({ code: shaderCode });
        pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: { module: shaderModule, entryPoint: 'main' },
        });
        WebGPUPipelineManager.registerPipeline(pipelineKey, pipeline);
    }

    // Create uniform buffer
    const uniformBuffer = createUniformBuffer(uniformData);

    // Create bind group
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: output.storage.buffer as GPUBuffer } },
            { binding: 1, resource: { buffer: uniformBuffer } },
        ],
    });

    // Execute shader
    const numWorkgroups = Math.ceil(windowLength / workgroupSize);
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
}

// ============================================================================
// Shader Builders
// ============================================================================

function buildGeneralizedCosineShader(dtype: DType, workgroupSize: number): string {
    const resolver = getGlobalDTypeResolver();
    const wgslType = resolver.getDescriptor(dtype).wgslStorageType;

    return `
struct WindowUniforms {
    numel: u32,
    denominator: f32,
    coeff0: f32,
    coeff1: f32,
    coeff2: f32,
    coeff3: f32,
    outputOffset: u32,
};

@group(0) @binding(0) var<storage, read_write> output: array<${wgslType}>;
@group(0) @binding(1) var<uniform> uniforms: WindowUniforms;

const PI: f32 = ${WGSL_CONSTANTS.PI};

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.numel) { return; }

    // Special case: window_length=1 returns [1.0] (PyTorch behavior)
    if (uniforms.numel == 1u) {
        output[uniforms.outputOffset + idx] = ${wgslType}(1.0);
        return;
    }

    let n = f32(idx);
    let angle = 2.0 * PI * n / uniforms.denominator;
    
    let val = uniforms.coeff0 
            - uniforms.coeff1 * cos(angle)
            + uniforms.coeff2 * cos(2.0 * angle)
            - uniforms.coeff3 * cos(3.0 * angle);
    
    output[uniforms.outputOffset + idx] = ${wgslType}(val);
}
`;
}

function buildLinearShader(dtype: DType, workgroupSize: number, windowType: 'bartlett' | 'triang'): string {
    const resolver = getGlobalDTypeResolver();
    const wgslType = resolver.getDescriptor(dtype).wgslStorageType;

    // Bartlett: w[n] = 1 - |2n/(M-1) - 1|
    // Triang: w[n] = 1 - |2(n - M/2)/M|
    const formula = windowType === 'bartlett'
        ? 'let val = 1.0 - abs(2.0 * n / uniforms.denominator - 1.0);'
        : 'let val = 1.0 - abs(2.0 * (n - uniforms.denominator / 2.0) / uniforms.denominator);';

    return `
struct WindowUniforms {
    numel: u32,
    denominator: f32,
    outputOffset: u32,
};

@group(0) @binding(0) var<storage, read_write> output: array<${wgslType}>;
@group(0) @binding(1) var<uniform> uniforms: WindowUniforms;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.numel) { return; }

    // Special case: window_length=1 returns [1.0] (PyTorch behavior)
    if (uniforms.numel == 1u) {
        output[uniforms.outputOffset + idx] = ${wgslType}(1.0);
        return;
    }

    let n = f32(idx);
    ${formula}
    
    output[uniforms.outputOffset + idx] = ${wgslType}(val);
}
`;
}

function buildKaiserShader(dtype: DType, workgroupSize: number): string {
    const resolver = getGlobalDTypeResolver();
    const wgslType = resolver.getDescriptor(dtype).wgslStorageType;

    return `
struct WindowUniforms {
    numel: u32,
    denominator: f32,
    beta: f32,
    i0_beta: f32,
    outputOffset: u32,
};

@group(0) @binding(0) var<storage, read_write> output: array<${wgslType}>;
@group(0) @binding(1) var<uniform> uniforms: WindowUniforms;

// Modified Bessel function I₀(x) - WGSL implementation
fn bessel_i0(x: f32) -> f32 {
    let ax = abs(x);
    var ans: f32;
    
    if (ax < 3.75) {
        let y = (x / 3.75) * (x / 3.75);
        ans = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
            + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))));
    } else {
        let y = 3.75 / ax;
        ans = (exp(ax) / sqrt(ax)) * (0.39894228 + y * (0.1328592e-1
            + y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2
            + y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1
            + y * 0.392377e-2))))))));
    }
    return ans;
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.numel) { return; }

    // Special case: window_length=1 returns [1.0] (PyTorch behavior)
    if (uniforms.numel == 1u) {
        output[uniforms.outputOffset + idx] = ${wgslType}(1.0);
        return;
    }

    let n = f32(idx);
    let center = uniforms.denominator / 2.0;
    let normalized_pos = (n - center) / center;
    let arg = uniforms.beta * sqrt(1.0 - normalized_pos * normalized_pos);
    
    let val = bessel_i0(arg) / uniforms.i0_beta;
    
    output[uniforms.outputOffset + idx] = ${wgslType}(val);
}
`;
}

function buildGaussianShader(dtype: DType, workgroupSize: number): string {
    const resolver = getGlobalDTypeResolver();
    const wgslType = resolver.getDescriptor(dtype).wgslStorageType;

    return `
struct WindowUniforms {
    numel: u32,
    denominator: f32,
    std: f32,
    outputOffset: u32,
};

@group(0) @binding(0) var<storage, read_write> output: array<${wgslType}>;
@group(0) @binding(1) var<uniform> uniforms: WindowUniforms;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.numel) { return; }

    // Special case: window_length=1 returns [1.0] (PyTorch behavior)
    if (uniforms.numel == 1u) {
        output[uniforms.outputOffset + idx] = ${wgslType}(1.0);
        return;
    }

    let n = f32(idx);
    let center = uniforms.denominator / 2.0;
    let sigma = uniforms.std * center;
    let normalized_dist = (n - center) / sigma;
    
    let val = exp(-0.5 * normalized_dist * normalized_dist);
    
    output[uniforms.outputOffset + idx] = ${wgslType}(val);
}
`;
}

function buildExponentialShader(dtype: DType, workgroupSize: number): string {
    const resolver = getGlobalDTypeResolver();
    const wgslType = resolver.getDescriptor(dtype).wgslStorageType;

    return `
struct WindowUniforms {
    numel: u32,
    denominator: f32,
    tau: f32,
    center: f32,
    outputOffset: u32,
};

@group(0) @binding(0) var<storage, read_write> output: array<${wgslType}>;
@group(0) @binding(1) var<uniform> uniforms: WindowUniforms;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.numel) { return; }

    // Special case: window_length=1 returns [1.0] (PyTorch behavior)
    if (uniforms.numel == 1u) {
        output[uniforms.outputOffset + idx] = ${wgslType}(1.0);
        return;
    }

    let n = f32(idx);
    let dist = abs(n - uniforms.center);
    
    let val = exp(-dist / uniforms.tau);
    
    output[uniforms.outputOffset + idx] = ${wgslType}(val);
}
`;
}

// ============================================================================
// Uniform Buffer Creators
// ============================================================================

function createGeneralizedCosineUniforms(
    windowLength: number,
    denominator: number,
    coeffs: [number, number, number, number],
    output: any
): ArrayBuffer {
    const resolver = getGlobalDTypeResolver();
    const bytesPerElement = resolver.getDescriptor(output.dtype).gpuBytesPerElement;

    const data = new ArrayBuffer(32);
    const u32View = new Uint32Array(data);
    const f32View = new Float32Array(data);

    u32View[0] = windowLength;
    f32View[1] = denominator;
    f32View[2] = coeffs[0];
    f32View[3] = coeffs[1];
    f32View[4] = coeffs[2];
    f32View[5] = coeffs[3];
    u32View[6] = output.offset / bytesPerElement;

    return data;
}

function createLinearUniforms(
    windowLength: number,
    denominator: number,
    output: any
): ArrayBuffer {
    const resolver = getGlobalDTypeResolver();
    const bytesPerElement = resolver.getDescriptor(output.dtype).gpuBytesPerElement;

    const data = new ArrayBuffer(16);
    const u32View = new Uint32Array(data);
    const f32View = new Float32Array(data);

    u32View[0] = windowLength;
    f32View[1] = denominator;
    u32View[2] = output.offset / bytesPerElement;

    return data;
}

function createKaiserUniforms(
    windowLength: number,
    denominator: number,
    beta: number,
    output: any
): ArrayBuffer {
    const resolver = getGlobalDTypeResolver();
    const bytesPerElement = resolver.getDescriptor(output.dtype).gpuBytesPerElement;

    // Pre-compute I₀(β) for normalization
    const i0_beta = computeBesselI0(beta);

    const data = new ArrayBuffer(20);
    const u32View = new Uint32Array(data);
    const f32View = new Float32Array(data);

    u32View[0] = windowLength;
    f32View[1] = denominator;
    f32View[2] = beta;
    f32View[3] = i0_beta;
    u32View[4] = output.offset / bytesPerElement;

    return data;
}

function createGaussianUniforms(
    windowLength: number,
    denominator: number,
    std: number,
    output: any
): ArrayBuffer {
    const resolver = getGlobalDTypeResolver();
    const bytesPerElement = resolver.getDescriptor(output.dtype).gpuBytesPerElement;

    const data = new ArrayBuffer(16);
    const u32View = new Uint32Array(data);
    const f32View = new Float32Array(data);

    u32View[0] = windowLength;
    f32View[1] = denominator;
    f32View[2] = std;
    u32View[3] = output.offset / bytesPerElement;

    return data;
}

function createExponentialUniforms(
    windowLength: number,
    denominator: number,
    tau: number,
    center: number | undefined,
    output: any
): ArrayBuffer {
    const resolver = getGlobalDTypeResolver();
    const bytesPerElement = resolver.getDescriptor(output.dtype).gpuBytesPerElement;

    // Default center is (M-1)/2
    const actualCenter = center ?? (denominator / 2.0);

    const data = new ArrayBuffer(20);
    const u32View = new Uint32Array(data);
    const f32View = new Float32Array(data);

    u32View[0] = windowLength;
    f32View[1] = denominator;
    f32View[2] = tau;
    f32View[3] = actualCenter;
    u32View[4] = output.offset / bytesPerElement;

    return data;
}
