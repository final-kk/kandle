/**
 * Triangular Shader Builder (工业级实现)
 * 
 * 生成 triu/tril 的 WGSL shader
 * 
 * 工业级特性：
 * - 输入使用 shape/strides/offset 计算物理地址，支持非连续访问
 * - 输出总是连续的 (新分配的 buffer)
 * 
 * 核心逻辑:
 * - 计算每个元素的 (row, col) 坐标 (最后两维)
 * - triu: 如果 col >= row + diagonal，保留原值，否则置零
 * - tril: 如果 col <= row + diagonal，保留原值，否则置零
 * 
 * 支持批量矩阵: (..., M, N)
 * 
 * 参考: PyTorch ATen/native/TensorShape.cpp
 */

import type { TriangularShaderParams } from './types';

/**
 * 构建 Triangular shader (工业级：strided 输入，contiguous 输出)
 */
export function buildTriangularShader(params: TriangularShaderParams): string {
    const {
        config,
        inputShape,
        inputStrides,
        inputOffset,
        diagonal,
        wgslType,
        workgroupSize,
    } = params;

    const ndim = inputShape.length;
    const numel = inputShape.reduce((a, b) => a * b, 1);

    // 最后两维是 (M, N)
    const M = ndim >= 2 ? inputShape[ndim - 2] : 1;
    const N = ndim >= 1 ? inputShape[ndim - 1] : 1;

    // 条件判断
    // PyTorch 定义:
    // triu(k): 保留 col - row >= k (即 col >= row + k) 的元素
    // tril(k): 保留 col - row <= k (即 col <= row + k) 的元素
    const condition = config.isUpper
        ? 'i32(col) >= i32(row) + DIAGONAL'  // triu: col >= row + k
        : 'i32(col) <= i32(row) + DIAGONAL'; // tril: col <= row + k

    // 生成 strided 索引计算代码
    const indexingCode = generateStridedIndexingCode(inputShape, inputStrides, inputOffset);

    return `
// ============================================================================
// Triangular Matrix: ${config.name} (工业级: Strided Input, Contiguous Output)
// Shape: [${inputShape.join(', ')}]
// Strides: [${inputStrides.join(', ')}]
// M=${M}, N=${N}, diagonal=${diagonal}
// Condition: ${config.isUpper ? 'col >= row + diagonal' : 'col <= row + diagonal'}
// ============================================================================

${indexingCode.constants}
const DIAGONAL: i32 = ${diagonal};
const M: u32 = ${M}u;
const N: u32 = ${N}u;
const NUMEL: u32 = ${numel}u;

@group(0) @binding(0) var<storage, read> input: array<${wgslType}>;
@group(0) @binding(1) var<storage, read_write> output: array<${wgslType}>;

${indexingCode.computePhysicalOffset}

@compute @workgroup_size(${workgroupSize})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    
    if (idx >= NUMEL) {
        return;
    }
    
    // 计算在当前 2D 切片中的位置 (row, col)
    let local_idx = idx % (M * N);
    let row = local_idx / N;
    let col = local_idx % N;
    
    // 计算输入的物理偏移 (strided 访问)
    let input_physical_offset = compute_physical_offset(idx);
    
    // 输出偏移 = 连续 idx
    let output_offset = idx;
    
    // 根据条件选择值
    var value: ${wgslType};
    if (${condition}) {
        value = input[input_physical_offset];
    } else {
        value = ${wgslType}(0);
    }
    
    output[output_offset] = value;
}
`;
}

// ============================================================================
// 辅助函数：Strided 索引计算
// ============================================================================

interface StridedIndexingCode {
    constants: string;
    computePhysicalOffset: string;
}

/**
 * 生成 strided 索引计算代码
 * 
 * 从 flat index (逻辑线性索引) 计算 strided 物理偏移
 * 这适用于 triangular 操作的情况：每个元素独立处理
 */
function generateStridedIndexingCode(
    shape: readonly number[],
    strides: readonly number[],
    offset: number
): StridedIndexingCode {
    const ndim = shape.length;

    // 生成常量
    const stridesConst = `const INPUT_STRIDES: array<i32, ${ndim}> = array<i32, ${ndim}>(${strides.join(', ')});`;
    const shapeConst = `const INPUT_SHAPE: array<u32, ${ndim}> = array<u32, ${ndim}>(${shape.join(', ')});`;
    const offsetConst = `const INPUT_OFFSET: i32 = ${offset};`;
    const ndimConst = `const NDIM: u32 = ${ndim}u;`;

    // 计算各维度的后缀积 (用于从 flat_idx 展开坐标)
    // suffix[i] = shape[i+1] * shape[i+2] * ... * shape[ndim-1]
    const suffixes: number[] = [];
    for (let i = 0; i < ndim; i++) {
        let suffix = 1;
        for (let j = i + 1; j < ndim; j++) {
            suffix *= shape[j];
        }
        suffixes.push(suffix);
    }
    const suffixesConst = `const SUFFIXES: array<u32, ${ndim}> = array<u32, ${ndim}>(${suffixes.join(', ')});`;

    // 生成 compute_physical_offset 函数
    // 从 flat_idx (逻辑线性索引) 计算 strided 物理偏移
    const computePhysicalOffset = `
// 从逻辑线性索引计算物理偏移 (strided 访问)
fn compute_physical_offset(flat_idx: u32) -> u32 {
    var offset: i32 = INPUT_OFFSET;
    var rem = flat_idx;
    
    // 展开 flat_idx 到各维度坐标，并累加 strided 偏移
    for (var d = 0u; d < NDIM; d++) {
        let coord = rem / SUFFIXES[d];
        rem = rem % SUFFIXES[d];
        offset += i32(coord) * INPUT_STRIDES[d];
    }
    
    return u32(offset);
}`;

    return {
        constants: [stridesConst, shapeConst, offsetConst, ndimConst, suffixesConst].join('\n'),
        computePhysicalOffset,
    };
}
