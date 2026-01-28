/**
 * MatMul Shader Builder
 * 
 * 生成 Tiled Matrix Multiplication 的 WGSL Shader
 * 
 * 参考 ONNX Runtime WebGPU 实现:
 * - 使用 workgroup shared memory 进行 tiling
 * - 支持 vec4 向量化加载
 * - 支持 transpose 标志
 * - 支持 batch 索引映射
 */

import { DType } from '@kandle/types';
import { Logger } from '@kandle/utils';
import { getGlobalDTypeResolver } from '../../base/DTypeResolver';
import { TileConfig, MatmulDispatchResult } from './types';
import {
    generateMmUniformStruct,
    generateBmmUniformStruct,
    generateBufferBindings,
    generateScalarSharedMemory,
    generateVec4SharedMemory,
    generateTileConstants,
    generateBmmConstants,
    generateBmmBufferBindings,
    generateBmmSharedMemory,
    generateBmmTileConstants,
} from './shaderTemplates';

const logger = new Logger('Matmul-Builder');

// ============================================================
// 类型映射
// ============================================================

function getWgslType(dtype: DType): string {
    const resolver = getGlobalDTypeResolver();
    return resolver.getDescriptor(dtype).wgslComputeType;
}

function getWgslStorageType(dtype: DType): string {
    const resolver = getGlobalDTypeResolver();
    return resolver.getDescriptor(dtype).wgslStorageType;
}

// ============================================================
// R6: FP16 Accumulator 优化
//
// 当使用 FP16 输入时，使用 FP32 累加器以提高精度
// 这对于长 K 维度的矩阵乘法非常重要，可以避免精度损失
// ============================================================

/**
 * 检查是否为复数类型
 */
function isComplexDtype(dtype: DType): boolean {
    return dtype === 'complex64' || dtype === 'complex128';
}

/**
 * 获取累加器类型
 * 
 * R6: 对于 FP16 输入，使用 FP32 累加器以提高精度
 * 这在 K 维度较大时可以显著减少精度损失
 * 
 * 对于复数类型，累加器始终是 vec2<f32>
 */
function getAccumulatorType(computeDtype: DType): string {
    // Complex types: always vec2<f32> (complex128 is downgraded to complex64 on GPU)
    if (isComplexDtype(computeDtype)) {
        return 'vec2<f32>';
    }
    // FP16 使用 FP32 累加，其他类型使用同类型累加
    if (computeDtype === 'float16') {
        return 'f32';
    }
    return getWgslType(computeDtype);
}

/**
 * 检查是否需要混合精度累加
 */
function needsMixedPrecision(computeDtype: DType): boolean {
    return computeDtype === 'float16';
}

/**
 * 生成复数乘法辅助函数 (cmul)
 * 
 * 复数乘法公式: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
 * 使用 vec2<f32> 表示复数: .x = real, .y = imag
 */
function generateCmulHelper(): string {
    return `
// Complex multiplication: (a.x + a.y*i) * (b.x + b.y*i) = (a.x*b.x - a.y*b.y) + (a.x*b.y + a.y*b.x)*i
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}
`;
}

// ============================================================
// GEMM 辅助函数
// ============================================================

/**
 * 生成 mm_readC 函数（读取 InputC 并处理广播）
 * 
 * 工业级实现：支持完整的 NumPy 广播语义
 * - InputC 形状可以是 [], [N], [1, N], [M, 1], [M, N]
 * - 当某个维度是 1 时，该维度会被广播
 */
function generateMmReadC(
    dataType: string,
    storageType: string,
    hasInputC: boolean
): string {
    if (!hasInputC) {
        // 如果没有 InputC，返回一个虚拟函数返回 0
        return `
fn mm_readC(row: u32, col: u32) -> ${dataType} {
    return ${dataType}(0.0);
}
`;
    }

    // 有 InputC - 需要处理广播
    // 使用 uniforms.c_shape_m 和 uniforms.c_shape_n 来处理广播
    // 当 c_shape_m == 1 时，所有 row 都读取 row 0 的数据
    // 当 c_shape_n == 1 时，所有 col 都读取 col 0 的数据
    return `
fn mm_readC(row: u32, col: u32) -> ${dataType} {
    if (row >= uniforms.M || col >= uniforms.N) {
        return ${dataType}(0.0);
    }
    // 处理广播：当 InputC 的某个维度是 1 时，使用模运算广播
    let actual_row = row % uniforms.c_shape_m;
    let actual_col = col % uniforms.c_shape_n;
    let idx = actual_row * uniforms.c_shape_n + actual_col;
    return ${dataType}(inputC[idx + uniforms.offset_c]);
}
`;
}

/**
 * 生成 mm_write 函数（GEMM 版本）
 * 
 * 三种优化变体：
 * - beta=0: 直接写入 alpha * value
 * - alpha=1, beta=1: 简化为 value + C
 * - general: beta * C + alpha * value
 * 
 * R6: 支持混合精度 - 累加器类型可能与数据类型不同
 */
function generateMmWrite(
    dataType: string,
    storageType: string,
    alpha: number,
    beta: number,
    hasInputC: boolean,
    accumulatorType?: string  // R6: 累加器类型（可选，用于混合精度）
): string {
    const isPureMatmul = beta === 0.0 || !hasInputC;
    const isSimpleAddmm = hasInputC && alpha === 1.0 && beta === 1.0;

    // R6: 确定累加器类型（如果未指定，使用数据类型）
    const accType = accumulatorType ?? dataType;
    const needsConversion = accType !== dataType;

    if (isPureMatmul) {
        // 快速路径：beta=0，不需要读取 C
        if (alpha === 1.0) {
            // 最简单：直接写入
            return `
fn mm_write(row: u32, col: u32, value: ${accType}) {
    if (row < uniforms.M && col < uniforms.N) {
        let idx = row * uniforms.N + col;
        output[idx + uniforms.offset_out] = ${storageType}(value);
    }
}
`;
        } else {
            // alpha != 1: 缩放后写入
            return `
fn mm_write(row: u32, col: u32, value: ${accType}) {
    if (row < uniforms.M && col < uniforms.N) {
        let idx = row * uniforms.N + col;
        let alpha_val = ${accType}(uniforms.alpha);
        output[idx + uniforms.offset_out] = ${storageType}(alpha_val * value);
    }
}
`;
        }
    } else if (isSimpleAddmm) {
        // 优化路径：alpha=1, beta=1 → result + C
        // R6: 需要将 C 值转换为累加器类型
        return `
fn mm_write(row: u32, col: u32, value: ${accType}) {
    if (row < uniforms.M && col < uniforms.N) {
        let idx = row * uniforms.N + col;
        let c_val = ${accType}(mm_readC(row, col));
        output[idx + uniforms.offset_out] = ${storageType}(value + c_val);
    }
}
`;
    } else {
        // 通用路径：beta * C + alpha * value
        return `
fn mm_write(row: u32, col: u32, value: ${accType}) {
    if (row < uniforms.M && col < uniforms.N) {
        let idx = row * uniforms.N + col;
        let alpha_val = ${accType}(uniforms.alpha);
        let beta_val = ${accType}(uniforms.beta);
        let c_val = ${accType}(mm_readC(row, col));
        let result = beta_val * c_val + alpha_val * value;
        output[idx + uniforms.offset_out] = ${storageType}(result);
    }
}
`;
    }
}

/**
 * 生成 mm_readC 函数（BMM 版本，带 batch 参数）
 * 
 * BMM 专用：读取 InputC 并处理 batch 和 2D 广播
 * 支持 M/N 维度的广播：当 c_shape_m 或 c_shape_n 为 1 时广播
 */
function generateBmmReadC(
    dataType: string,
    storageType: string,
    hasInputC: boolean,
    hasBatchDim: boolean  // InputC 是否有 batch 维度
): string {
    if (!hasInputC) {
        return `
fn mm_readC(batch: u32, row: u32, col: u32) -> ${dataType} {
    return ${dataType}(0.0);
}
`;
    }

    if (hasBatchDim) {
        // InputC 有 batch 维度，需要使用 batch 转换器
        // 同时处理 M/N 维度的广播
        return `
fn mm_readC(batch: u32, row: u32, col: u32) -> ${dataType} {
    if (row >= uniforms.M || col >= uniforms.N) {
        return ${dataType}(0.0);
    }
    // InputC 有 batch 维度，使用转换器处理广播
    let batchC = convertBatchIdxC(batch);
    // 处理 M/N 维度广播
    let actual_row = row % uniforms.c_shape_m;
    let actual_col = col % uniforms.c_shape_n;
    let batchOffset = batchC * uniforms.c_shape_m * uniforms.c_shape_n;
    let idx = batchOffset + actual_row * uniforms.c_shape_n + actual_col;
    return ${dataType}(inputC[idx + uniforms.offset_c]);
}
`;
    } else {
        // InputC 是 2D，所有 batch 共享同一个矩阵
        // 同时处理 M/N 维度的广播
        return `
fn mm_readC(batch: u32, row: u32, col: u32) -> ${dataType} {
    if (row >= uniforms.M || col >= uniforms.N) {
        return ${dataType}(0.0);
    }
    // InputC 是 2D，所有 batch 共享（batch 参数被忽略）
    // 处理 M/N 维度广播
    let actual_row = row % uniforms.c_shape_m;
    let actual_col = col % uniforms.c_shape_n;
    let idx = actual_row * uniforms.c_shape_n + actual_col;
    return ${dataType}(inputC[idx + uniforms.offset_c]);
}
`;
    }
}

/**
 * 生成 mm_write 函数（BMM 版本，带 batch 参数）
 * 
 * BMM 专用：三种优化变体
 * 
 * R6: 支持混合精度 - 累加器类型可能与数据类型不同
 */
function generateBmmWrite(
    dataType: string,
    storageType: string,
    alpha: number,
    beta: number,
    hasInputC: boolean,
    accumulatorType?: string  // R6: 累加器类型（可选，用于混合精度）
): string {
    const isPureMatmul = beta === 0.0 || !hasInputC;
    const isSimpleAddmm = hasInputC && alpha === 1.0 && beta === 1.0;

    // R6: 确定累加器类型
    const accType = accumulatorType ?? dataType;

    if (isPureMatmul) {
        if (alpha === 1.0) {
            return `
fn mm_write(batch: u32, row: u32, col: u32, value: ${accType}) {
    if (batch < uniforms.batch_size && row < uniforms.M && col < uniforms.N) {
        let batchOffset = batch * MN;
        let idx = batchOffset + row * uniforms.N + col;
        output[idx + uniforms.offset_out] = ${storageType}(value);
    }
}
`;
        } else {
            return `
fn mm_write(batch: u32, row: u32, col: u32, value: ${accType}) {
    if (batch < uniforms.batch_size && row < uniforms.M && col < uniforms.N) {
        let batchOffset = batch * MN;
        let idx = batchOffset + row * uniforms.N + col;
        let alpha_val = ${accType}(uniforms.alpha);
        output[idx + uniforms.offset_out] = ${storageType}(alpha_val * value);
    }
}
`;
        }
    } else if (isSimpleAddmm) {
        // R6: 需要将 C 值转换为累加器类型
        return `
fn mm_write(batch: u32, row: u32, col: u32, value: ${accType}) {
    if (batch < uniforms.batch_size && row < uniforms.M && col < uniforms.N) {
        let batchOffset = batch * MN;
        let idx = batchOffset + row * uniforms.N + col;
        let c_val = ${accType}(mm_readC(batch, row, col));
        output[idx + uniforms.offset_out] = ${storageType}(value + c_val);
    }
}
`;
    } else {
        return `
fn mm_write(batch: u32, row: u32, col: u32, value: ${accType}) {
    if (batch < uniforms.batch_size && row < uniforms.M && col < uniforms.N) {
        let batchOffset = batch * MN;
        let idx = batchOffset + row * uniforms.N + col;
        let alpha_val = ${accType}(uniforms.alpha);
        let beta_val = ${accType}(uniforms.beta);
        let c_val = ${accType}(mm_readC(batch, row, col));
        let result = beta_val * c_val + alpha_val * value;
        output[idx + uniforms.offset_out] = ${storageType}(result);
    }
}
`;
    }
}

// ============================================================
// Shader 生成器
// ============================================================

/**
 * 生成 MM Shader (2D @ 2D)
 * 
 * 最基础的 tiled matmul，支持 GEMM (alpha, beta)
 * 
 * R6: 自动使用高精度累加器 (FP16 -> FP32)
 */
export function buildMmShader(
    config: MatmulDispatchResult,
    tileConfig: TileConfig
): string {
    const { M, K, N, transposeA, transposeB, computeDtype, alpha, beta, inputC } = config;
    const { workPerThread, workgroupSize, tileInner, useVec4 } = tileConfig;

    const dataType = getWgslType(computeDtype);
    const storageType = getWgslStorageType(computeDtype);

    // R6: 获取累加器类型
    const accumulatorType = getAccumulatorType(computeDtype);

    const rowPerThread = workPerThread[1];
    const colPerThread = workPerThread[0];

    const tileAOuter = workgroupSize[1] * rowPerThread;
    const tileBOuter = workgroupSize[0] * colPerThread;

    const hasInputC = inputC !== undefined && beta !== 0.0;

    // 检查是否需要 f16 扩展
    const needsF16 = computeDtype === 'float16' && getGlobalDTypeResolver().supportsNativeF16;
    const enableF16 = needsF16 ? 'enable f16;\n' : '';

    if (useVec4) {
        return buildMmShaderVec4(
            M, K, N,
            transposeA, transposeB,
            dataType, storageType, accumulatorType,
            rowPerThread, colPerThread,
            tileAOuter, tileBOuter, tileInner,
            workgroupSize,
            enableF16,
            alpha,
            beta,
            hasInputC
        );
    } else {
        return buildMmShaderScalar(
            M, K, N,
            transposeA, transposeB,
            dataType, storageType, accumulatorType,
            rowPerThread, colPerThread,
            tileAOuter, tileBOuter, tileInner,
            workgroupSize,
            enableF16,
            alpha,
            beta,
            hasInputC
        );
    }
}

/**
 * Vec4 优化版本
 * 
 * R6: 支持混合精度累加器
 */
function buildMmShaderVec4(
    M: number, K: number, N: number,
    transposeA: boolean, transposeB: boolean,
    dataType: string, storageType: string, accumulatorType: string,  // R6: added
    rowPerThread: number, colPerThread: number,
    tileAOuter: number, tileBOuter: number, tileInner: number,
    workgroupSize: [number, number, number],
    enableF16: string,
    alpha: number,
    beta: number,
    hasInputC: boolean
): string {
    const innerElementSize = transposeA ? 4 : tileInner / workgroupSize[0];
    const rowPerThreadB = tileInner / workgroupSize[1];

    // 生成 GEMM 专用函数 (R6: 传递累加器类型)
    const mmReadC = generateMmReadC(dataType, storageType, hasInputC);
    const mmWrite = generateMmWrite(dataType, storageType, alpha, beta, hasInputC, accumulatorType);

    // 使用模板生成 Uniform 结构和 Buffer 绑定
    const uniformStruct = generateMmUniformStruct(hasInputC);
    const bufferBindings = generateBufferBindings(storageType, hasInputC);
    const sharedMemory = generateVec4SharedMemory(dataType, tileAOuter, tileBOuter, tileInner);
    const tileConstants = generateTileConstants(rowPerThread, colPerThread, tileInner, M, K, N);

    // R6: 累加器使用高精度类型
    const accVec4Type = `vec4<${accumulatorType}>`;
    // Vec4 类型用于 shader 中的向量操作
    const vec4Type = `vec4<${dataType}>`;

    const shader = `
${enableF16}
// Uniforms
${uniformStruct}

${bufferBindings}

${sharedMemory}

${tileConstants}

// Helper functions
fn mm_readA(row: u32, col: u32) -> ${dataType} {
    if (row >= M_val || col >= K_val) {
        return ${dataType}(0.0);
    }
    ${transposeA ?
            `let idx = col * M_val + row;` :
            `let idx = row * K_val + col;`}
    return ${dataType}(inputA[idx + uniforms.offset_a]);
}

fn mm_readB(row: u32, col: u32) -> ${dataType} {
    if (row >= K_val || col >= N_val) {
        return ${dataType}(0.0);
    }
    ${transposeB ?
            `let idx = col * K_val + row;` :
            `let idx = row * N_val + col;`}
    return ${dataType}(inputB[idx + uniforms.offset_b]);
}

${mmReadC}

${mmWrite}

@compute @workgroup_size(${workgroupSize[0]}, ${workgroupSize[1]}, ${workgroupSize[2]})
fn main(
    @builtin(local_invocation_id) localId: vec3<u32>,
    @builtin(global_invocation_id) globalId: vec3<u32>,
    @builtin(workgroup_id) workgroupId: vec3<u32>
) {
    let localRow = i32(localId.y);
    let localCol = i32(localId.x);
    let globalRow = i32(globalId.y) * ${rowPerThread};
    let globalCol = i32(globalId.x) * ${colPerThread};
    let globalRowStart = i32(workgroupId.y) * ${tileAOuter};
    let globalColStart = i32(workgroupId.x) * ${tileBOuter};

    let numTiles = (K_val + tileInner - 1u) / tileInner;
    var kStart = 0u;

    // R6: 累加器使用高精度类型
    var acc: array<${accVec4Type}, ${rowPerThread}>;
    for (var i = 0u; i < ${rowPerThread}u; i = i + 1u) {
        acc[i] = ${accVec4Type}(0.0);
    }

    // Loop over tiles
    for (var t = 0u; t < numTiles; t = t + 1u) {
        // Load tile A into shared memory
        for (var innerRow = 0u; innerRow < ${rowPerThread}u; innerRow = innerRow + 1u) {
            let inputRow = u32(localRow) * ${rowPerThread}u + innerRow;
            let inputCol = u32(localCol);
            if (inputRow < ${tileAOuter}u && inputCol * 4u < tileInner) {
                let aRow = u32(globalRowStart) + inputRow;
                let aCol = kStart + inputCol * 4u;
                mm_Asub[inputCol * 4u][inputRow / 4u][inputRow % 4u] = mm_readA(aRow, aCol);
                mm_Asub[inputCol * 4u + 1u][inputRow / 4u][inputRow % 4u] = mm_readA(aRow, aCol + 1u);
                mm_Asub[inputCol * 4u + 2u][inputRow / 4u][inputRow % 4u] = mm_readA(aRow, aCol + 2u);
                mm_Asub[inputCol * 4u + 3u][inputRow / 4u][inputRow % 4u] = mm_readA(aRow, aCol + 3u);
            }
        }

        // Load tile B into shared memory
        for (var innerRow = 0u; innerRow < ${rowPerThreadB}u; innerRow = innerRow + 1u) {
            let inputRow = u32(localRow) * ${rowPerThreadB}u + innerRow;
            let inputCol = u32(localCol);
            if (inputRow < tileInner && inputCol * 4u < ${tileBOuter}u) {
                let bRow = kStart + inputRow;
                let bCol = u32(globalColStart) + inputCol * 4u;
                mm_Bsub[inputRow][inputCol] = ${vec4Type}(
                    mm_readB(bRow, bCol),
                    mm_readB(bRow, bCol + 1u),
                    mm_readB(bRow, bCol + 2u),
                    mm_readB(bRow, bCol + 3u)
                );
            }
        }

        kStart = kStart + tileInner;
        workgroupBarrier();

        // Compute - R6: 累加使用高精度
        for (var k = 0u; k < tileInner; k = k + 1u) {
            let BCached = ${accVec4Type}(mm_Bsub[k][u32(localCol)]);
            for (var innerRow = 0u; innerRow < ${rowPerThread}u; innerRow = innerRow + 1u) {
                let ACached = ${accumulatorType}(mm_Asub[k][(u32(localRow) * ${rowPerThread}u + innerRow) / 4u][(u32(localRow) * ${rowPerThread}u + innerRow) % 4u]);
                acc[innerRow] = acc[innerRow] + BCached * ACached;
            }
        }

        workgroupBarrier();
    }

    // Write results
    for (var innerRow = 0u; innerRow < ${rowPerThread}u; innerRow = innerRow + 1u) {
        let outRow = u32(globalRow) + innerRow;
        for (var innerCol = 0u; innerCol < ${colPerThread}u; innerCol = innerCol + 1u) {
            let outCol = u32(globalCol) + innerCol;
            mm_write(outRow, outCol, acc[innerRow][innerCol]);
        }
    }
}
`;

    logger.debug('Generated Vec4 Shader with R6 mixed-precision accumulator');
    return shader;
}

/**
 * 标量版本 (fallback)
 * 
 * R6: 支持混合精度累加器
 */
function buildMmShaderScalar(
    M: number, K: number, N: number,
    transposeA: boolean, transposeB: boolean,
    dataType: string, storageType: string, accumulatorType: string,  // R6: added
    rowPerThread: number, colPerThread: number,
    tileAOuter: number, tileBOuter: number, tileInner: number,
    workgroupSize: [number, number, number],
    enableF16: string,
    alpha: number,
    beta: number,
    hasInputC: boolean
): string {
    const rowPerThreadA = tileAOuter / workgroupSize[1];
    const colPerThreadA = tileInner / workgroupSize[0];
    const rowPerThreadB = tileInner / workgroupSize[1];

    // 生成 GEMM 专用函数 (R6: 传递累加器类型)
    const mmReadC = generateMmReadC(dataType, storageType, hasInputC);
    const mmWrite = generateMmWrite(dataType, storageType, alpha, beta, hasInputC, accumulatorType);

    // 使用模板生成 Uniform 结构和 Buffer 绑定
    const uniformStruct = generateMmUniformStruct(hasInputC);
    const bufferBindings = generateBufferBindings(storageType, hasInputC);
    const sharedMemory = generateScalarSharedMemory(dataType, tileAOuter, tileBOuter, tileInner);
    const tileConstants = generateTileConstants(rowPerThread, colPerThread, tileInner);

    const shader = `
${enableF16}
// Uniforms
${uniformStruct}

${bufferBindings}

${sharedMemory}

${tileConstants}

// Helper functions
fn mm_readA(row: u32, col: u32) -> ${dataType} {
    if (row >= uniforms.M || col >= uniforms.K) {
        return ${dataType}(0.0);
    }
    ${transposeA ?
            `let idx = col * uniforms.M + row;` :
            `let idx = row * uniforms.K + col;`}
    return ${dataType}(inputA[idx + uniforms.offset_a]);
}

fn mm_readB(row: u32, col: u32) -> ${dataType} {
    if (row >= uniforms.K || col >= uniforms.N) {
        return ${dataType}(0.0);
    }
    ${transposeB ?
            `let idx = col * uniforms.K + row;` :
            `let idx = row * uniforms.N + col;`}
    return ${dataType}(inputB[idx + uniforms.offset_b]);
}

${mmReadC}

${mmWrite}

@compute @workgroup_size(${workgroupSize[0]}, ${workgroupSize[1]}, ${workgroupSize[2]})
fn main(
    @builtin(local_invocation_id) localId: vec3<u32>,
    @builtin(global_invocation_id) globalId: vec3<u32>,
    @builtin(workgroup_id) workgroupId: vec3<u32>
) {
    let tileRow = i32(localId.y) * ${rowPerThread};
    let tileCol = i32(localId.x) * ${colPerThread};
    let globalRow = i32(globalId.y) * ${rowPerThread};
    let globalCol = i32(globalId.x) * ${colPerThread};
    let globalRowStart = i32(workgroupId.y) * ${tileAOuter};
    let globalColStart = i32(workgroupId.x) * ${tileBOuter};

    let numTiles = (uniforms.K + tileInner - 1u) / tileInner;
    var kStart = 0u;

    // R6: 累加器使用高精度类型
    var acc: array<array<${accumulatorType}, ${colPerThread}>, ${rowPerThread}>;
    for (var i = 0u; i < ${rowPerThread}u; i = i + 1u) {
        for (var j = 0u; j < ${colPerThread}u; j = j + 1u) {
            acc[i][j] = ${accumulatorType}(0.0);
        }
    }

    let tileRowA = i32(localId.y) * ${rowPerThreadA};
    let tileColA = i32(localId.x) * ${colPerThreadA};
    let tileRowB = i32(localId.y) * ${rowPerThreadB};

    // Loop over tiles
    for (var t = 0u; t < numTiles; t = t + 1u) {
        // Load tile A into shared memory
        for (var innerRow = 0u; innerRow < ${rowPerThreadA}u; innerRow = innerRow + 1u) {
            for (var innerCol = 0u; innerCol < ${colPerThreadA}u; innerCol = innerCol + 1u) {
                let inputRow = u32(tileRowA) + innerRow;
                let inputCol = u32(tileColA) + innerCol;
                mm_Asub[inputRow][inputCol] = mm_readA(u32(globalRowStart) + inputRow, kStart + inputCol);
            }
        }

        // Load tile B into shared memory
        for (var innerRow = 0u; innerRow < ${rowPerThreadB}u; innerRow = innerRow + 1u) {
            for (var innerCol = 0u; innerCol < ${colPerThread}u; innerCol = innerCol + 1u) {
                let inputRow = u32(tileRowB) + innerRow;
                let inputCol = u32(tileCol) + innerCol;
                mm_Bsub[inputRow][inputCol] = mm_readB(kStart + inputRow, u32(globalColStart) + inputCol);
            }
        }

        kStart = kStart + tileInner;
        workgroupBarrier();

        // Compute - R6: 累加使用高精度
        var BCached: array<${accumulatorType}, ${colPerThread}>;
        for (var k = 0u; k < tileInner; k = k + 1u) {
            for (var inner = 0u; inner < ${colPerThread}u; inner = inner + 1u) {
                BCached[inner] = ${accumulatorType}(mm_Bsub[k][u32(tileCol) + inner]);
            }

            for (var innerRow = 0u; innerRow < ${rowPerThread}u; innerRow = innerRow + 1u) {
                let ACached = ${accumulatorType}(mm_Asub[u32(tileRow) + innerRow][k]);
                for (var innerCol = 0u; innerCol < ${colPerThread}u; innerCol = innerCol + 1u) {
                    acc[innerRow][innerCol] = acc[innerRow][innerCol] + ACached * BCached[innerCol];
                }
            }
        }

        workgroupBarrier();
    }

    // Write results
    for (var innerRow = 0u; innerRow < ${rowPerThread}u; innerRow = innerRow + 1u) {
        for (var innerCol = 0u; innerCol < ${colPerThread}u; innerCol = innerCol + 1u) {
            mm_write(u32(globalRow) + innerRow, u32(globalCol) + innerCol, acc[innerRow][innerCol]);
        }
    }
}
`;

    logger.debug('Generated Scalar Shader with R6 mixed-precision accumulator');
    return shader;
}

/**
 * 生成用于 batch 索引转换的 WGSL 代码
 * 
 * 将输出的线性 batch 索引转换为输入的线性 batch 索引
 * 处理广播：如果输入的某个 batch 维度是 1，则该维度索引始终为 0
 * 
 * @param outputBatchShape 输出的 batch 形状（广播后）
 * @param inputBatchShape 输入的 batch 形状（可能有 1 需要广播）
 * @param funcName 生成的函数名
 * @returns WGSL 函数代码
 */
function generateBatchIndexConverter(
    outputBatchShape: readonly number[],
    inputBatchShape: readonly number[],
    funcName: string
): string {
    const outputRank = outputBatchShape.length;
    const inputRank = inputBatchShape.length;

    // 如果都是空的（无 batch 维度）
    if (outputRank === 0 && inputRank === 0) {
        return `
fn ${funcName}(outputBatchIdx: u32) -> u32 {
    return 0u;
}
`;
    }

    // 计算输出 batch 的 strides（用于分解线性索引到多维索引）
    const outputStrides: number[] = [];
    let stride = 1;
    for (let i = outputRank - 1; i >= 0; i--) {
        outputStrides.unshift(stride);
        stride *= outputBatchShape[i];
    }

    // 计算输入 batch 的 strides（用于将多维索引合成线性索引）
    const inputStrides: number[] = [];
    stride = 1;
    for (let i = inputRank - 1; i >= 0; i--) {
        inputStrides.unshift(stride);
        stride *= inputBatchShape[i];
    }

    // 生成 WGSL 代码
    // 思路：
    // 1. 将 outputBatchIdx 分解为多维索引 [d0, d1, ..., dn]
    // 2. 对每个维度，如果 inputBatchShape[i] == 1，则该维度索引为 0（广播）
    // 3. 将调整后的多维索引合成 inputBatchIdx

    let code = `
fn ${funcName}(outputBatchIdx: u32) -> u32 {
    var remaining = outputBatchIdx;
    var inputIdx = 0u;
`;

    // 右对齐：输入可能比输出少维度，从右边匹配
    // 例如：output = [2, 3, 4], input = [3, 1] -> input 对应 output 的后两维
    const dimOffset = outputRank - inputRank;

    for (let i = 0; i < outputRank; i++) {
        const outputDim = outputBatchShape[i];
        const outputStride = outputStrides[i];

        // 计算这一维的索引
        if (i < outputRank - 1) {
            code += `    let idx${i} = remaining / ${outputStride}u;\n`;
            code += `    remaining = remaining % ${outputStride}u;\n`;
        } else {
            code += `    let idx${i} = remaining;\n`;
        }

        // 检查这一维是否对应到 input（右对齐）
        const inputDimIndex = i - dimOffset;
        if (inputDimIndex >= 0 && inputDimIndex < inputRank) {
            const inputDim = inputBatchShape[inputDimIndex];
            const inputStride = inputStrides[inputDimIndex];

            if (inputDim === 1) {
                // 广播：这一维不贡献到 inputIdx（索引为 0）
                code += `    // dim ${i}: input broadcasts (size=1)\n`;
            } else {
                // 正常维度：贡献到 inputIdx
                code += `    inputIdx = inputIdx + idx${i} * ${inputStride}u;\n`;
            }
        }
        // 如果 inputDimIndex < 0，说明 output 在左边有额外的维度，input 隐式广播
    }

    code += `    return inputIdx;
}
`;

    return code;
}

/**
 * 计算一个 batch shape 的总大小
 */
function computeBatchSize(batchShape: readonly number[]): number {
    return batchShape.reduce((acc, dim) => acc * dim, 1);
}

/**
 * 生成 BMM Shader (≥3D @ ≥3D，带 batch 广播)
 * 
 * 关键：正确处理 batch 维度的广播
 * - 输出的 batch 索引是线性的 (0 to batchSize-1)
 * - 需要映射到 A 和 B 各自的 batch 索引
 * 
 * R6: 自动使用高精度累加器 (FP16 -> FP32)
 */
export function buildBmmShader(
    config: MatmulDispatchResult,
    tileConfig: TileConfig,
    batchShapeA: readonly number[],
    batchShapeB: readonly number[],
    batchShapeC?: readonly number[]  // 新增：InputC 的 batch shape
): string {
    const { M, K, N, batchShape, batchSize, transposeA, transposeB, computeDtype, alpha, beta, inputC } = config;
    const { workPerThread, workgroupSize, tileInner } = tileConfig;

    const dataType = getWgslType(computeDtype);
    const storageType = getWgslStorageType(computeDtype);

    // R6: 获取累加器类型
    const accumulatorType = getAccumulatorType(computeDtype);

    const rowPerThread = workPerThread[1];
    const colPerThread = workPerThread[0];

    const tileAOuter = workgroupSize[1] * rowPerThread;
    const tileBOuter = workgroupSize[0] * colPerThread;

    const needsF16 = computeDtype === 'float16' && getGlobalDTypeResolver().supportsNativeF16;
    const enableF16 = needsF16 ? 'enable f16;\n' : '';

    const rowPerThreadA = tileAOuter / workgroupSize[1];
    const colPerThreadA = tileInner / workgroupSize[0];
    const rowPerThreadB = tileInner / workgroupSize[1];

    // 计算 A 和 B 各自的 batch size（用于索引计算）
    const batchSizeA = computeBatchSize(batchShapeA);
    const batchSizeB = computeBatchSize(batchShapeB);

    // GEMM 支持
    const hasInputC = inputC !== undefined && beta !== 0.0;

    // 复数类型检测 (complex64/complex128)
    const isComplex = isComplexDtype(computeDtype);
    const cmulHelper = isComplex ? generateCmulHelper() : '';

    // 生成 batch 索引转换函数
    const batchConverterA = generateBatchIndexConverter(batchShape, batchShapeA, 'convertBatchIdxA');
    const batchConverterB = generateBatchIndexConverter(batchShape, batchShapeB, 'convertBatchIdxB');

    // InputC 的 batch 转换器（如果 InputC 存在）
    let batchConverterC = '';
    if (hasInputC && batchShapeC && batchShapeC.length > 0) {
        batchConverterC = generateBatchIndexConverter(batchShape, batchShapeC, 'convertBatchIdxC');
    }

    // 生成 GEMM 辅助函数（BMM 版本）(R6: 传递累加器类型)
    const bmmReadC = generateBmmReadC(dataType, storageType, hasInputC, !!(batchShapeC && batchShapeC.length > 0));
    const bmmWrite = generateBmmWrite(dataType, storageType, alpha, beta, hasInputC, accumulatorType);

    // 使用模板生成 Uniform 结构、Buffer 绑定、Shared Memory 和常量
    const uniformStruct = generateBmmUniformStruct(hasInputC);
    const bufferBindings = generateBmmBufferBindings(storageType, hasInputC);
    const sharedMemory = generateBmmSharedMemory(dataType, tileAOuter, tileBOuter, tileInner);
    const tileConstants = generateBmmTileConstants(rowPerThread, colPerThread, tileInner, M, K, N);

    const shader = `
${enableF16}
// Uniforms
${uniformStruct}

${bufferBindings}

${sharedMemory}

${tileConstants}

${cmulHelper}

// Batch index conversion functions
${batchConverterA}
${batchConverterB}
${batchConverterC}

// Helper functions with batch offset and broadcasting
// 真正的 4D 索引计算：支持任意 strided 输入 (如 KV Cache slice)
// 
// 索引公式:
//   对于 4D tensor [d0, d1, d2, d3]:
//   idx = offset + d0 * strides[0] + d1 * strides[1] + d2 * strides[2] + d3 * strides[3]
//
// 在 BMM 中，我们将 batch 展平为线性索引，需要还原:
//   - outputBatch 是展平后的 batch 索引
//   - convertBatchIdxA/B 处理 batch 广播
//   - 对于 4D: batchIdx 对应 d0, headIdx 对应 d1, row 对应 d2, col 对应 d3
//   - 对于 3D: batchIdx 对应 d0/d1, row 对应 d2, col 对应 d3
//
// strides_a/b 是完整的 4D strides (已 padding 到 4 维):
//   - 2D [M, K] -> strides = [0, 0, K, 1]
//   - 3D [B, M, K] -> strides = [0, M*K, K, 1]  
//   - 4D [B, H, M, K] -> strides = [H*M*K, M*K, K, 1] (或任意非连续模式)

fn mm_readA(outputBatch: u32, row: u32, col: u32) -> ${dataType} {
    if (row >= uniforms.M || col >= uniforms.K) {
        return ${dataType}(0.0);
    }
    let batchA = convertBatchIdxA(outputBatch);
    
    // 真正的 4D 索引计算
    // 对于 4D: batchA 包含了 batch 和 head 信息，需要解开
    // 对于 3D: strides_a[0] = 0, batchA 直接用于 strides_a[1]
    // 对于 2D: strides_a[0] = strides_a[1] = 0
    //
    // 通用公式: idx = offset + (batch 部分) + row * strides[2] + col * strides[3]
    // batch 部分通过 batch 转换器已经处理了广播
    ${transposeA
            ? `// transposeA: 逻辑 row (M) <-> 物理 col (d2), 逻辑 col (K) <-> 物理 row (d3)
    let idx = uniforms.offset_a 
            + batchA * uniforms.strides_a[1]  // 如果有 batch/head 维度
            + col * uniforms.strides_a[2]     // 物理上的 row 位置
            + row * uniforms.strides_a[3];    // 物理上的 col 位置`
            : `// 正常顺序: row 对应 d2, col 对应 d3
    let idx = uniforms.offset_a 
            + batchA * uniforms.strides_a[1]  // 如果有 batch/head 维度
            + row * uniforms.strides_a[2]     // row 位置
            + col * uniforms.strides_a[3];    // col 位置`
        }
    return ${dataType}(inputA[idx]);
}

fn mm_readB(outputBatch: u32, row: u32, col: u32) -> ${dataType} {
    if (row >= uniforms.K || col >= uniforms.N) {
        return ${dataType}(0.0);
    }
    let batchB = convertBatchIdxB(outputBatch);
    
    // 真正的 4D 索引计算
    ${transposeB
            ? `// transposeB: 逻辑 row (K) <-> 物理 col (d2), 逻辑 col (N) <-> 物理 row (d3)
    let idx = uniforms.offset_b 
            + batchB * uniforms.strides_b[1]  // 如果有 batch/head 维度
            + col * uniforms.strides_b[2]     // 物理上的 row 位置
            + row * uniforms.strides_b[3];    // 物理上的 col 位置`
            : `// 正常顺序: row 对应 d2, col 对应 d3
    let idx = uniforms.offset_b 
            + batchB * uniforms.strides_b[1]  // 如果有 batch/head 维度
            + row * uniforms.strides_b[2]     // row 位置
            + col * uniforms.strides_b[3];    // col 位置`
        }
    return ${dataType}(inputB[idx]);
}


${bmmReadC}

${bmmWrite}

@compute @workgroup_size(${workgroupSize[0]}, ${workgroupSize[1]}, ${workgroupSize[2]})
fn main(
    @builtin(local_invocation_id) localId: vec3<u32>,
    @builtin(global_invocation_id) globalId: vec3<u32>,
    @builtin(workgroup_id) workgroupId: vec3<u32>
) {
    let batch = workgroupId.z;

    let tileRow = i32(localId.y) * ${rowPerThread};
    let tileCol = i32(localId.x) * ${colPerThread};
    let globalRow = i32(globalId.y) * ${rowPerThread};
    let globalCol = i32(globalId.x) * ${colPerThread};
    let globalRowStart = i32(workgroupId.y) * ${tileAOuter};
    let globalColStart = i32(workgroupId.x) * ${tileBOuter};

    let numTiles = (uniforms.K + tileInner - 1u) / tileInner;
    var kStart = 0u;

    // R6: 累加器使用高精度类型
    var acc: array<array<${accumulatorType}, ${colPerThread}>, ${rowPerThread}>;
    for (var i = 0u; i < ${rowPerThread}u; i = i + 1u) {
        for (var j = 0u; j < ${colPerThread}u; j = j + 1u) {
            acc[i][j] = ${isComplex ? 'vec2<f32>(0.0, 0.0)' : `${accumulatorType}(0.0)`};
        }
    }

    let tileRowA = i32(localId.y) * ${rowPerThreadA};
    let tileColA = i32(localId.x) * ${colPerThreadA};
    let tileRowB = i32(localId.y) * ${rowPerThreadB};

    // Loop over tiles
    for (var t = 0u; t < numTiles; t = t + 1u) {
        // Load tile A
        for (var innerRow = 0u; innerRow < ${rowPerThreadA}u; innerRow = innerRow + 1u) {
            for (var innerCol = 0u; innerCol < ${colPerThreadA}u; innerCol = innerCol + 1u) {
                let inputRow = u32(tileRowA) + innerRow;
                let inputCol = u32(tileColA) + innerCol;
                mm_Asub[inputRow][inputCol] = mm_readA(batch, u32(globalRowStart) + inputRow, kStart + inputCol);
            }
        }

        // Load tile B
        for (var innerRow = 0u; innerRow < ${rowPerThreadB}u; innerRow = innerRow + 1u) {
            for (var innerCol = 0u; innerCol < ${colPerThread}u; innerCol = innerCol + 1u) {
                let inputRow = u32(tileRowB) + innerRow;
                let inputCol = u32(tileCol) + innerCol;
                mm_Bsub[inputRow][inputCol] = mm_readB(batch, kStart + inputRow, u32(globalColStart) + inputCol);
            }
        }

        kStart = kStart + tileInner;
        workgroupBarrier();

        // Compute - R6: 累加使用高精度
        var BCached: array<${accumulatorType}, ${colPerThread}>;
        for (var k = 0u; k < tileInner; k = k + 1u) {
            for (var inner = 0u; inner < ${colPerThread}u; inner = inner + 1u) {
                BCached[inner] = ${accumulatorType}(mm_Bsub[k][u32(tileCol) + inner]);
            }

            for (var innerRow = 0u; innerRow < ${rowPerThread}u; innerRow = innerRow + 1u) {
                let ACached = ${accumulatorType}(mm_Asub[u32(tileRow) + innerRow][k]);
                for (var innerCol = 0u; innerCol < ${colPerThread}u; innerCol = innerCol + 1u) {
                    ${isComplex ? 'acc[innerRow][innerCol] = acc[innerRow][innerCol] + cmul(ACached, BCached[innerCol]);' : 'acc[innerRow][innerCol] = acc[innerRow][innerCol] + ACached * BCached[innerCol];'}
                }
            }
        }

        workgroupBarrier();
    }

    // Write results
    for (var innerRow = 0u; innerRow < ${rowPerThread}u; innerRow = innerRow + 1u) {
        for (var innerCol = 0u; innerCol < ${colPerThread}u; innerCol = innerCol + 1u) {
            mm_write(batch, u32(globalRow) + innerRow, u32(globalCol) + innerCol, acc[innerRow][innerCol]);
        }
    }
}
`;

    logger.debug(`Generated BMM Shader with R6 mixed-precision: output=${JSON.stringify(batchShape)}, A=${JSON.stringify(batchShapeA)}, B=${JSON.stringify(batchShapeB)}`);
    return shader;
}
