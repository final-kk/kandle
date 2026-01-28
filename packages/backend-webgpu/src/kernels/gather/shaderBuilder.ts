/**
 * Gather Kernels - WGSL Shader Builder
 * 
 * 生成 index_select 等操作的 WGSL shader
 * 
 * 工业级实现：原生支持 strided (非连续) 输入
 * 通过 shape/strides/offset 计算物理地址，无需预先克隆
 */

import type { DType } from '@kandle/types';
import type { IndexSelectShaderParams } from './types';

/**
 * 获取 WGSL 数据类型
 */
function wgslType(dtype: DType): string {
    switch (dtype) {
        case 'float32': return 'f32';
        case 'float64': return 'f32';  // WebGPU 不支持 f64，需要在高层处理
        case 'int32': return 'i32';
        case 'int64': return 'i32';    // WebGPU 不支持 i64
        case 'uint32': return 'u32';
        case 'int16': return 'i32';    // 使用 i32 兼容
        case 'uint16': return 'u32';
        case 'int8': return 'i32';
        case 'uint8': return 'u32';
        case 'bool': return 'u32';
        default: return 'f32';
    }
}

/**
 * 生成 index_select WGSL shader (工业级 strided 实现)
 * 
 * 算法:
 * 1. 对于输出中的每个位置 idx，计算其 N 维坐标
 * 2. 从索引张量获取 dim 维度的实际源索引 (支持 strided index)
 * 3. 使用输入的 strides 计算源物理地址 (支持 strided input)
 * 4. 写入连续输出
 * 
 * 关键：使用 strides 进行物理地址计算，原生支持非连续内存布局
 */
export function buildIndexSelectShader(params: IndexSelectShaderParams): string {
    const {
        inputShape, inputStrides, inputOffset,
        indexLength, indexStride, indexOffset,
        outputShape, outputStrides,
        dim, dtype, outputSize
    } = params;
    const ndim = inputShape.length;
    const wType = wgslType(dtype);

    // 生成 shape/stride 常量
    const inputShapeStr = `array<u32, ${ndim}>(${inputShape.join(', ')})`;
    // 输入 strides 可能是非标准的 (来自 transpose/permute)，使用 i32 支持负步幅
    const inputStridesStr = `array<i32, ${ndim}>(${inputStrides.join(', ')})`;
    const outputShapeStr = `array<u32, ${ndim}>(${outputShape.join(', ')})`;
    // 输出总是连续的，strides 用于 unflatten
    const outputStridesStr = `array<u32, ${ndim}>(${outputStrides.join(', ')})`;

    return `
// index_select kernel (Strided Implementation)
// dim = ${dim}, ndim = ${ndim}, outputSize = ${outputSize}
// Supports non-contiguous input via strides

@group(0) @binding(0) var<storage, read> input: array<${wType}>;
@group(0) @binding(1) var<storage, read> index: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<${wType}>;

const INPUT_SHAPE: array<u32, ${ndim}> = ${inputShapeStr};
const INPUT_STRIDES: array<i32, ${ndim}> = ${inputStridesStr};
const INPUT_OFFSET: u32 = ${inputOffset}u;
const INDEX_STRIDE: i32 = ${indexStride};
const INDEX_OFFSET: u32 = ${indexOffset}u;
const OUTPUT_SHAPE: array<u32, ${ndim}> = ${outputShapeStr};
const OUTPUT_STRIDES: array<u32, ${ndim}> = ${outputStridesStr};
const NDIM: u32 = ${ndim}u;
const DIM: u32 = ${dim}u;
const OUTPUT_SIZE: u32 = ${outputSize}u;

// 将线性索引转换为 N 维坐标 (用于输出，总是连续的)
fn unflatten(idx: u32, strides: ptr<function, array<u32, ${ndim}>>) -> array<u32, ${ndim}> {
    var coords: array<u32, ${ndim}>;
    var remaining = idx;
    for (var i = 0u; i < NDIM; i++) {
        coords[i] = remaining / (*strides)[i];
        remaining = remaining % (*strides)[i];
    }
    return coords;
}

// 将 N 维坐标转换为物理偏移量 (使用 strided input strides)
// 这是支持非连续内存的关键：使用实际的 strides 而非假设连续
fn compute_input_offset(coords: ptr<function, array<u32, ${ndim}>>) -> u32 {
    var offset: i32 = i32(INPUT_OFFSET);
    for (var i = 0u; i < NDIM; i++) {
        offset += i32((*coords)[i]) * INPUT_STRIDES[i];
    }
    return u32(offset);
}

// 计算 index 张量中的物理位置 (支持 strided 1D index)
fn compute_index_offset(logical_idx: u32) -> u32 {
    return INDEX_OFFSET + u32(i32(logical_idx) * INDEX_STRIDE);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let outIdx = gid.x;
    if (outIdx >= OUTPUT_SIZE) {
        return;
    }

    // 计算输出坐标 (输出是连续的)
    var outStrides = OUTPUT_STRIDES;
    let outCoords = unflatten(outIdx, &outStrides);

    // 从索引张量获取源索引 (支持 strided index)
    let indexPhysicalPos = compute_index_offset(outCoords[DIM]);
    let indexVal = u32(index[indexPhysicalPos]);

    // 构造源坐标（将 dim 维度替换为 indexVal）
    var srcCoords: array<u32, ${ndim}>;
    for (var i = 0u; i < NDIM; i++) {
        if (i == DIM) {
            srcCoords[i] = indexVal;
        } else {
            srcCoords[i] = outCoords[i];
        }
    }

    // 计算源物理偏移量 (使用 strided strides)
    let srcOffset = compute_input_offset(&srcCoords);

    // 读取并写入
    output[outIdx] = input[srcOffset];
}
`;
}
