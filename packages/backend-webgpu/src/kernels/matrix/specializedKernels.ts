/**
 * Specialized Kernels for MV and Dot Operations
 * 
 * 专用的 Matrix-Vector 和 Dot Product 内核
 * 
 * 优点：
 * - MV: 使用 1D workgroup，每个线程处理一行，更好的内存访问模式
 * - Dot: 使用 tree reduction 并行归约，高效利用 GPU 并行性
 * 
 * 这些内核比复用 MM kernel 更高效
 */

import { DType } from '@kandle/types';
import { getGlobalDTypeResolver } from '../../base/DTypeResolver';

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

/**
 * 检查是否为复数类型
 */
function isComplexDtype(dtype: DType): boolean {
    return dtype === 'complex64' || dtype === 'complex128';
}

/**
 * 生成复数乘法辅助函数 (cmul)
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
// Matrix-Vector Kernel
// ============================================================

export interface MvShaderConfig {
    M: number;          // 矩阵行数 (输出长度)
    K: number;          // 矩阵列数 / 向量长度 (公共维度)
    transposeA: boolean;
    dtype: DType;
    alpha: number;
    beta: number;
    hasInputC: boolean;
    // === 工业级 Strided 支持 ===
    /** A 的 row stride (沿 M 维度移动一单位需要的元素数) */
    strideA_row: number;
    /** A 的 col stride (沿 K 维度移动一单位需要的元素数) */
    strideA_col: number;
    /** B (vector) 的 stride (1D tensor) */
    strideB: number;
    /** C 的 stride (如果有) */
    strideC: number;
}

/**
 * Matrix-Vector Shader 配置
 * 
 * 工作组配置：
 * - workgroupSize: [256, 1, 1] - 每个 workgroup 处理 256 行
 * - 每个线程处理一行的所有列
 * - 适合 K 较小的场景（K <= 1024）
 */
export interface MvTileConfig {
    workgroupSize: number;  // 通常为 256
    rowsPerWorkgroup: number; // 等于 workgroupSize
}

export function selectMvTileConfig(M: number, K: number): MvTileConfig {
    // 对于小矩阵使用较小的 workgroup
    const workgroupSize = M < 64 ? 64 : 256;
    return {
        workgroupSize,
        rowsPerWorkgroup: workgroupSize,
    };
}

/**
 * 构建 Matrix-Vector shader (工业级: Strided 支持)
 * 
 * 设计：
 * - 每个线程处理一行 (M 方向并行)
 * - 使用串行循环处理 K 维度
 * - 支持非连续输入：通过 strides 计算物理地址
 * - 适合 K 较小的场景
 */
export function buildMvShader(config: MvShaderConfig, tileConfig: MvTileConfig): string {
    const { dtype, alpha, beta, hasInputC, strideA_row, strideA_col, strideB, strideC } = config;
    const { workgroupSize } = tileConfig;

    const dataType = getWgslType(dtype);
    const storageType = getWgslStorageType(dtype);

    const needsF16 = dtype === 'float16' && getGlobalDTypeResolver().supportsNativeF16;
    const enableF16 = needsF16 ? 'enable f16;\n' : '';

    // 复数类型检测
    const isComplex = isComplexDtype(dtype);
    const cmulHelper = isComplex ? generateCmulHelper() : '';
    // 复数的乘法需要用 cmul，加法可以直接用 +
    const multiplyExpr = isComplex
        ? (a: string, b: string) => `cmul(${a}, ${b})`
        : (a: string, b: string) => `${a} * ${b}`;
    const zeroValue = isComplex ? 'vec2<f32>(0.0, 0.0)' : `${dataType}(0.0)`;

    // GEMM 支持
    const isPureMatmul = !hasInputC || beta === 0.0;
    const isSimpleAddmm = hasInputC && alpha === 1.0 && beta === 1.0;

    // === Strided 访问常量 ===
    // 编译时常量，避免 uniform overhead
    const stridedConstants = `
// Strides (工业级: 支持非连续访问)
const STRIDE_A_ROW: i32 = ${strideA_row};  // A 沿 M 移动的 stride
const STRIDE_A_COL: i32 = ${strideA_col};  // A 沿 K 移动的 stride
const STRIDE_B: i32 = ${strideB};          // B vector 的 stride
const STRIDE_C: i32 = ${strideC};          // C vector 的 stride (如果有)
`;

    // 写入函数根据 GEMM 变体生成
    let writeLogic: string;
    if (isPureMatmul) {
        if (alpha === 1.0) {
            writeLogic = `output[row + uniforms.offset_out] = ${storageType}(acc);`;
        } else {
            writeLogic = `output[row + uniforms.offset_out] = ${storageType}(${dataType}(uniforms.alpha) * acc);`;
        }
    } else if (isSimpleAddmm) {
        writeLogic = `
        let c_idx = i32(uniforms.offset_c) + i32(row) * STRIDE_C;
        let c_val = ${dataType}(inputC[c_idx]);
        output[row + uniforms.offset_out] = ${storageType}(acc + c_val);`;
    } else {
        writeLogic = `
        let c_idx = i32(uniforms.offset_c) + i32(row) * STRIDE_C;
        let c_val = ${dataType}(inputC[c_idx]);
        let result = ${dataType}(uniforms.beta) * c_val + ${dataType}(uniforms.alpha) * acc;
        output[row + uniforms.offset_out] = ${storageType}(result);`;
    }

    // Uniform 结构
    const uniformStruct = hasInputC ? `
struct Uniforms {
    M: u32,
    K: u32,
    offset_a: u32,
    offset_b: u32,
    offset_c: u32,
    offset_out: u32,
    alpha: f32,
    beta: f32,
}
` : `
struct Uniforms {
    M: u32,
    K: u32,
    offset_a: u32,
    offset_b: u32,
    offset_out: u32,
    alpha: f32,
    beta: f32,
}
`;

    // Buffer 绑定
    const bufferBindings = hasInputC ? `
@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> inputA: array<${storageType}>;
@group(0) @binding(2) var<storage, read> inputB: array<${storageType}>;
@group(0) @binding(3) var<storage, read_write> output: array<${storageType}>;
@group(0) @binding(4) var<storage, read> inputC: array<${storageType}>;
` : `
@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> inputA: array<${storageType}>;
@group(0) @binding(2) var<storage, read> inputB: array<${storageType}>;
@group(0) @binding(3) var<storage, read_write> output: array<${storageType}>;
`;

    return `
${enableF16}
${cmulHelper}
${stridedConstants}
${uniformStruct}

${bufferBindings}

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let row = globalId.x;
    
    if (row >= uniforms.M) {
        return;
    }

    // 累加器
    var acc = ${zeroValue};

    // 循环处理 K 维度
    // 使用 strided 访问: A[row, k] = A[offset_a + row * stride_row + k * stride_col]
    for (var k = 0u; k < uniforms.K; k = k + 1u) {
        // Strided A 访问：offset + row * stride_row + k * stride_col
        let a_idx = i32(uniforms.offset_a) + i32(row) * STRIDE_A_ROW + i32(k) * STRIDE_A_COL;
        // Strided B 访问：offset + k * stride
        let b_idx = i32(uniforms.offset_b) + i32(k) * STRIDE_B;
        
        let a_val = ${dataType}(inputA[a_idx]);
        let b_val = ${dataType}(inputB[b_idx]);
        acc = acc + ${multiplyExpr('a_val', 'b_val')};
    }

    // 写入结果 (输出总是连续的)
    ${writeLogic}
}
`;
}

// ============================================================
// Dot Product Kernel
// ============================================================

export interface DotShaderConfig {
    K: number;          // 向量长度
    dtype: DType;
    // === 工业级 Strided 支持 ===
    strideA: number;    // A 向量的 stride
    strideB: number;    // B 向量的 stride
}

/**
 * Dot Product Tile 配置
 * 
 * 使用 tree reduction:
 * - 第一阶段：每个线程处理多个元素，累加到 shared memory
 * - 第二阶段：shared memory 内并行归约
 */
export interface DotTileConfig {
    workgroupSize: number;      // 通常为 256
    elementsPerThread: number;  // 每个线程处理的元素数
}

export function selectDotTileConfig(K: number): DotTileConfig {
    // 工作组大小
    const workgroupSize = 256;

    // 每个线程处理的元素数 (确保能覆盖整个向量)
    const elementsPerThread = Math.max(1, Math.ceil(K / workgroupSize));

    return {
        workgroupSize,
        elementsPerThread,
    };
}

/**
 * 构建 Dot Product shader (工业级: Strided 支持)
 * 
 * 设计：
 * - 使用 tree reduction 实现并行归约
 * - 每个线程先处理多个元素，累加到 shared memory
 * - 然后在 shared memory 内进行并行归约
 * - 支持非连续输入：通过 strides 计算物理地址
 * - 最终结果写入输出 buffer
 */
export function buildDotShader(config: DotShaderConfig, tileConfig: DotTileConfig): string {
    const { dtype, strideA, strideB } = config;
    const { workgroupSize, elementsPerThread } = tileConfig;

    const dataType = getWgslType(dtype);
    const storageType = getWgslStorageType(dtype);

    const needsF16 = dtype === 'float16' && getGlobalDTypeResolver().supportsNativeF16;
    const enableF16 = needsF16 ? 'enable f16;\n' : '';

    // 复数类型检测
    const isComplex = isComplexDtype(dtype);
    const cmulHelper = isComplex ? generateCmulHelper() : '';
    const zeroValue = isComplex ? 'vec2<f32>(0.0, 0.0)' : `${dataType}(0.0)`;
    const multiplyExpr = isComplex
        ? (a: string, b: string) => `cmul(${a}, ${b})`
        : (a: string, b: string) => `${a} * ${b}`;

    return `
${enableF16}
${cmulHelper}
// Strides (工业级: 支持非连续访问)
const STRIDE_A: i32 = ${strideA};
const STRIDE_B: i32 = ${strideB};

struct Uniforms {
    K: u32,
    offset_a: u32,
    offset_b: u32,
    offset_out: u32,
}

@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> inputA: array<${storageType}>;
@group(0) @binding(2) var<storage, read> inputB: array<${storageType}>;
@group(0) @binding(3) var<storage, read_write> output: array<${storageType}>;

// Shared memory for reduction
var<workgroup> sharedData: array<${dataType}, ${workgroupSize}>;

@compute @workgroup_size(${workgroupSize}, 1, 1)
fn main(
    @builtin(local_invocation_id) localId: vec3<u32>,
    @builtin(workgroup_id) workgroupId: vec3<u32>
) {
    let tid = localId.x;
    
    // 阶段1：每个线程处理多个元素 (使用 strided 访问)
    var sum = ${zeroValue};
    for (var i = 0u; i < ${elementsPerThread}u; i = i + 1u) {
        let idx = tid + i * ${workgroupSize}u;
        if (idx < uniforms.K) {
            // Strided 访问: offset + idx * stride
            let a_idx = i32(uniforms.offset_a) + i32(idx) * STRIDE_A;
            let b_idx = i32(uniforms.offset_b) + i32(idx) * STRIDE_B;
            let a_val = ${dataType}(inputA[a_idx]);
            let b_val = ${dataType}(inputB[b_idx]);
            sum = sum + ${multiplyExpr('a_val', 'b_val')};
        }
    }
    
    // 存入 shared memory
    sharedData[tid] = sum;
    workgroupBarrier();
    
    // 阶段2：Tree reduction
    // 每次迭代，活跃线程数减半
    for (var stride = ${workgroupSize}u / 2u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            sharedData[tid] = sharedData[tid] + sharedData[tid + stride];
        }
        workgroupBarrier();
    }
    
    // 线程 0 写入最终结果
    if (tid == 0u) {
        output[uniforms.offset_out] = ${storageType}(sharedData[0]);
    }
}
`;
}

// ============================================================
// MV Uniform Buffer 
// ============================================================

export interface MvUniformBufferParams {
    M: number;
    K: number;
    offsetA: number;
    offsetB: number;
    offsetC?: number;
    offsetOut: number;
    alpha: number;
    beta: number;
}

export function createMvUniformBuffer(params: MvUniformBufferParams, hasInputC: boolean): ArrayBuffer {
    if (hasInputC) {
        // 8 个 u32/f32 字段
        const buffer = new ArrayBuffer(32);
        const view = new DataView(buffer);
        let offset = 0;
        view.setUint32(offset, params.M, true); offset += 4;
        view.setUint32(offset, params.K, true); offset += 4;
        view.setUint32(offset, params.offsetA, true); offset += 4;
        view.setUint32(offset, params.offsetB, true); offset += 4;
        view.setUint32(offset, params.offsetC ?? 0, true); offset += 4;
        view.setUint32(offset, params.offsetOut, true); offset += 4;
        view.setFloat32(offset, params.alpha, true); offset += 4;
        view.setFloat32(offset, params.beta, true);
        return buffer;
    } else {
        // 7 个 u32/f32 字段，需要 padding 到 32 字节对齐
        const buffer = new ArrayBuffer(32);
        const view = new DataView(buffer);
        let offset = 0;
        view.setUint32(offset, params.M, true); offset += 4;
        view.setUint32(offset, params.K, true); offset += 4;
        view.setUint32(offset, params.offsetA, true); offset += 4;
        view.setUint32(offset, params.offsetB, true); offset += 4;
        view.setUint32(offset, params.offsetOut, true); offset += 4;
        view.setFloat32(offset, params.alpha, true); offset += 4;
        view.setFloat32(offset, params.beta, true);
        return buffer;
    }
}

// ============================================================
// Dot Uniform Buffer
// ============================================================

export interface DotUniformBufferParams {
    K: number;
    offsetA: number;
    offsetB: number;
    offsetOut: number;
}

export function createDotUniformBuffer(params: DotUniformBufferParams): ArrayBuffer {
    // 4 个 u32 字段
    const buffer = new ArrayBuffer(16);
    const view = new DataView(buffer);
    view.setUint32(0, params.K, true);
    view.setUint32(4, params.offsetA, true);
    view.setUint32(8, params.offsetB, true);
    view.setUint32(12, params.offsetOut, true);
    return buffer;
}

// ============================================================
// Dispatch 计算
// ============================================================

export function calculateMvDispatch(M: number, tileConfig: MvTileConfig): [number, number, number] {
    const numWorkgroups = Math.ceil(M / tileConfig.workgroupSize);
    return [numWorkgroups, 1, 1];
}

export function calculateDotDispatch(): [number, number, number] {
    // Dot product 只需要一个 workgroup
    return [1, 1, 1];
}
