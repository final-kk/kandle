/**
 * ArgMax/ArgMin Shader Builder
 * 
 * 生成 argmax/argmin 归约的 WGSL shader
 * 
 * 与标准归约不同：
 * - 同时跟踪值和索引
 * - 输出只有索引 (int32)
 * - 使用 struct { val, idx } 模式
 */

import { ITensorIterator } from '@kandle/types';
import { Logger } from '@kandle/utils';
import { getStorageType, generateLoadSnippet } from '../../shader/ShaderSnippets';
import { getComputeType, generateCastSnippet } from '../../base/dtype';
import { getGlobalDTypeResolver } from '../../base/DTypeResolver';
import { WGSL_CONSTANTS } from '../../base/dtype';

const logger = new Logger('Arg-ShaderBuilder');

/**
 * argmax/argmin 操作配置
 */
interface ArgOpConfig {
    compare: (a: string, b: string) => string;  // 返回 true 如果 a 应该替换 b
    initialValue: (computeType: string) => string;
}

const ARG_OPS: Record<string, ArgOpConfig> = {
    'argmax': {
        compare: (a, b) => `${a} > ${b}`,
        initialValue: (t) => {
            if (t === 'f32' || t === 'f16') {
                return `${t}(${WGSL_CONSTANTS.NEG_FLT_MAX})`;
            } else if (t === 'i32') {
                return `i32(${WGSL_CONSTANTS.INT_MIN})`;
            } else if (t === 'u32') {
                return 'u32(0)';
            }
            return `${t}(-1e38)`;
        },
    },
    'argmin': {
        compare: (a, b) => `${a} < ${b}`,
        initialValue: (t) => {
            if (t === 'f32' || t === 'f16') {
                return `${t}(${WGSL_CONSTANTS.FLT_MAX})`;
            } else if (t === 'i32') {
                return `i32(${WGSL_CONSTANTS.INT_MAX})`;
            } else if (t === 'u32') {
                return `u32(${WGSL_CONSTANTS.UINT_MAX})`;
            }
            return `${t}(1e38)`;
        },
    },
};

/**
 * 构建 argmax/argmin 维度归约 shader
 */
export function buildArgDimReductionShader(
    iter: ITensorIterator,
    dispatchKey: 'argmax' | 'argmin',
    workgroupSize: number
): string {
    const opConfig = ARG_OPS[dispatchKey];
    if (!opConfig) {
        throw new Error(`Unknown arg operation: ${dispatchKey}`);
    }

    const input = iter.input(0);
    const output = iter.output();
    const resolver = getGlobalDTypeResolver();
    const computeType = getComputeType(iter.computeDtype);
    const storageTypeInput = getStorageType(input.dtype);
    // 输出是 int32 (WebGPU 不支持 int64)
    const storageTypeOutput = 'array<i32>';
    const elemTypeInput = resolver.getDescriptor(input.dtype).wgslStorageType;

    const loaderInput = generateLoadSnippet('input', input.dtype);

    const needsF16 = input.dtype === 'float16';
    const enableF16 = needsF16 && resolver.supportsNativeF16 ? 'enable f16;\n' : '';

    // 生成 tree reduction 代码
    const numSteps = Math.log2(workgroupSize);
    let treeReductionCode = '';
    for (let step = 0; step < numSteps; step++) {
        const stride = workgroupSize >> (step + 1);
        treeReductionCode += `
        workgroupBarrier();
        if (tid < ${stride}u) {
            let other_idx = tid + ${stride}u;
            if (${opConfig.compare('shared_vals[other_idx]', 'shared_vals[tid]')}) {
                shared_vals[tid] = shared_vals[other_idx];
                shared_idxs[tid] = shared_idxs[other_idx];
            }
        }
        `;
    }

    return `
${enableF16}
// Dimensional ArgReduction Shader: ${dispatchKey}
// Output shape: [${iter.outputShape.join(', ')}], Reduction shape: [${iter.reductionShape.join(', ')}]

struct Uniforms {
    output_numel: u32,
    reduction_numel: u32,
    output_rank: u32,
    reduction_rank: u32,
    output_shape0: vec4<u32>,
    output_shape1: vec4<u32>,
    reduction_shape0: vec4<u32>,
    reduction_shape1: vec4<u32>,
    input_parallel_strides0: vec4<u32>,
    input_parallel_strides1: vec4<u32>,
    input_reduction_strides0: vec4<u32>,
    input_reduction_strides1: vec4<u32>,
    output_strides0: vec4<u32>,
    output_strides1: vec4<u32>,
    input_offset: u32,
    output_offset: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: ${storageTypeInput};
@group(0) @binding(2) var<storage, read_write> output: ${storageTypeOutput};

var<workgroup> shared_vals: array<${computeType}, ${workgroupSize}>;
var<workgroup> shared_idxs: array<u32, ${workgroupSize}>;

${loaderInput.code}

fn get_output_shape(dim: u32) -> u32 {
    if (dim < 4u) { return uniforms.output_shape0[dim]; }
    else { return uniforms.output_shape1[dim - 4u]; }
}

fn get_reduction_shape(dim: u32) -> u32 {
    if (dim < 4u) { return uniforms.reduction_shape0[dim]; }
    else { return uniforms.reduction_shape1[dim - 4u]; }
}

fn get_input_parallel_stride(dim: u32) -> u32 {
    if (dim < 4u) { return uniforms.input_parallel_strides0[dim]; }
    else { return uniforms.input_parallel_strides1[dim - 4u]; }
}

fn get_input_reduction_stride(dim: u32) -> u32 {
    if (dim < 4u) { return uniforms.input_reduction_strides0[dim]; }
    else { return uniforms.input_reduction_strides1[dim - 4u]; }
}

fn get_output_stride(dim: u32) -> u32 {
    if (dim < 4u) { return uniforms.output_strides0[dim]; }
    else { return uniforms.output_strides1[dim - 4u]; }
}

fn compute_parallel_offset(output_flat_idx: u32) -> u32 {
    var offset: u32 = 0u;
    var remaining = output_flat_idx;
    for (var d: i32 = i32(uniforms.output_rank) - 1; d >= 0; d = d - 1) {
        let dim = u32(d);
        let dim_size = get_output_shape(dim);
        let coord = remaining % dim_size;
        remaining = remaining / dim_size;
        offset = offset + coord * get_input_parallel_stride(dim);
    }
    return offset;
}

fn compute_reduction_offset(reduction_flat_idx: u32) -> u32 {
    var offset: u32 = 0u;
    var remaining = reduction_flat_idx;
    for (var d: i32 = i32(uniforms.reduction_rank) - 1; d >= 0; d = d - 1) {
        let dim = u32(d);
        let dim_size = get_reduction_shape(dim);
        let coord = remaining % dim_size;
        remaining = remaining / dim_size;
        offset = offset + coord * get_input_reduction_stride(dim);
    }
    return offset;
}

fn compute_output_offset(output_flat_idx: u32) -> u32 {
    var offset: u32 = 0u;
    var remaining = output_flat_idx;
    for (var d: i32 = i32(uniforms.output_rank) - 1; d >= 0; d = d - 1) {
        let dim = u32(d);
        let dim_size = get_output_shape(dim);
        let coord = remaining % dim_size;
        remaining = remaining / dim_size;
        offset = offset + coord * get_output_stride(dim);
    }
    return offset;
}

@compute @workgroup_size(${workgroupSize})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let output_idx = wid.x;
    
    if (output_idx >= uniforms.output_numel) {
        return;
    }
    
    let parallel_base = compute_parallel_offset(output_idx);
    
    // 初始化为极值
    var best_val = ${opConfig.initialValue(computeType)};
    var best_idx: u32 = 0u;
    
    // 每个线程处理多个归约元素
    for (var r = tid; r < uniforms.reduction_numel; r = r + ${workgroupSize}u) {
        let reduction_offset = compute_reduction_offset(r);
        let input_idx = uniforms.input_offset + parallel_base + reduction_offset;
        let raw_val = ${loaderInput.funcName}(input_idx);
        let val = ${generateCastSnippet('raw_val', elemTypeInput, computeType)};
        
        if (${opConfig.compare('val', 'best_val')}) {
            best_val = val;
            best_idx = r;
        }
    }
    
    shared_vals[tid] = best_val;
    shared_idxs[tid] = best_idx;
    
    // Tree reduction
    ${treeReductionCode}
    
    // 线程 0 写结果
    if (tid == 0u) {
        let out_idx = uniforms.output_offset + compute_output_offset(output_idx);
        output[out_idx] = i32(shared_idxs[0]);
    }
}
`;
}

/**
 * 构建全局 argmax/argmin shader
 */
export function buildArgGlobalReductionShader(
    iter: ITensorIterator,
    dispatchKey: 'argmax' | 'argmin',
    workgroupSize: number,
    isContiguous: boolean,
    rank: number
): string {
    const opConfig = ARG_OPS[dispatchKey];
    if (!opConfig) {
        throw new Error(`Unknown arg operation: ${dispatchKey}`);
    }

    const input = iter.input(0);
    const resolver = getGlobalDTypeResolver();
    const computeType = getComputeType(iter.computeDtype);
    const storageTypeInput = getStorageType(input.dtype);
    const storageTypeOutput = 'array<i32>';
    const elemTypeInput = resolver.getDescriptor(input.dtype).wgslStorageType;

    const loaderInput = generateLoadSnippet('input', input.dtype);

    const needsF16 = input.dtype === 'float16';
    const enableF16 = needsF16 && resolver.supportsNativeF16 ? 'enable f16;\n' : '';

    // 生成 tree reduction 代码
    const numSteps = Math.log2(workgroupSize);
    let treeReductionCode = '';
    for (let step = 0; step < numSteps; step++) {
        const stride = workgroupSize >> (step + 1);
        treeReductionCode += `
        workgroupBarrier();
        if (tid < ${stride}u) {
            let other_idx = tid + ${stride}u;
            if (${opConfig.compare('shared_vals[other_idx]', 'shared_vals[tid]')}) {
                shared_vals[tid] = shared_vals[other_idx];
                shared_idxs[tid] = shared_idxs[other_idx];
            }
        }
        `;
    }

    if (isContiguous) {
        return `
${enableF16}
// Global ArgReduction (CONTIGUOUS): ${dispatchKey}

struct Uniforms {
    numel: u32,
    input_offset: u32,
    output_offset: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: ${storageTypeInput};
@group(0) @binding(2) var<storage, read_write> output: ${storageTypeOutput};

var<workgroup> shared_vals: array<${computeType}, ${workgroupSize}>;
var<workgroup> shared_idxs: array<u32, ${workgroupSize}>;

${loaderInput.code}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    
    // 每个线程累积多个元素
    var best_val = ${opConfig.initialValue(computeType)};
    var best_idx: u32 = 0u;
    
    for (var i = tid; i < uniforms.numel; i = i + ${workgroupSize}u) {
        let raw_val = ${loaderInput.funcName}(i + uniforms.input_offset);
        let val = ${generateCastSnippet('raw_val', elemTypeInput, computeType)};
        
        if (${opConfig.compare('val', 'best_val')}) {
            best_val = val;
            best_idx = i;
        }
    }
    
    shared_vals[tid] = best_val;
    shared_idxs[tid] = best_idx;
    
    // Tree reduction
    ${treeReductionCode}
    
    // 线程 0 写结果
    if (tid == 0u) {
        output[uniforms.output_offset] = i32(shared_idxs[0]);
    }
}
`;
    } else {
        // Strided (non-contiguous) version
        return `
${enableF16}
// Global ArgReduction (STRIDED): ${dispatchKey}, Rank: ${rank}

struct Uniforms {
    numel: u32,
    rank: u32,
    input_offset: u32,
    output_offset: u32,
    shape0: vec4<u32>,
    shape1: vec4<u32>,
    strides0: vec4<u32>,
    strides1: vec4<u32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: ${storageTypeInput};
@group(0) @binding(2) var<storage, read_write> output: ${storageTypeOutput};

var<workgroup> shared_vals: array<${computeType}, ${workgroupSize}>;
var<workgroup> shared_idxs: array<u32, ${workgroupSize}>;

${loaderInput.code}

fn get_shape(dim: u32) -> u32 {
    if (dim < 4u) { return uniforms.shape0[dim]; }
    else { return uniforms.shape1[dim - 4u]; }
}

fn get_stride(dim: u32) -> u32 {
    if (dim < 4u) { return uniforms.strides0[dim]; }
    else { return uniforms.strides1[dim - 4u]; }
}

fn compute_strided_offset(flat_idx: u32) -> u32 {
    var offset: u32 = 0u;
    var remaining = flat_idx;
    for (var d: i32 = i32(uniforms.rank) - 1; d >= 0; d = d - 1) {
        let dim = u32(d);
        let dim_size = get_shape(dim);
        let coord = remaining % dim_size;
        remaining = remaining / dim_size;
        offset = offset + coord * get_stride(dim);
    }
    return offset;
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    
    var best_val = ${opConfig.initialValue(computeType)};
    var best_idx: u32 = 0u;
    
    for (var i = tid; i < uniforms.numel; i = i + ${workgroupSize}u) {
        let physical_offset = compute_strided_offset(i);
        let raw_val = ${loaderInput.funcName}(physical_offset + uniforms.input_offset);
        let val = ${generateCastSnippet('raw_val', elemTypeInput, computeType)};
        
        if (${opConfig.compare('val', 'best_val')}) {
            best_val = val;
            best_idx = i;
        }
    }
    
    shared_vals[tid] = best_val;
    shared_idxs[tid] = best_idx;
    
    // Tree reduction
    ${treeReductionCode}
    
    // 线程 0 写结果
    if (tid == 0u) {
        output[uniforms.output_offset] = i32(shared_idxs[0]);
    }
}
`;
    }
}
