/**
 * MatMul Shader Templates
 * 
 * 提取公共的 WGSL 代码模板，减少 builder.ts 中的重复
 * 
 * 模块化设计：
 * - Uniform 结构模板
 * - Buffer 绑定模板
 * - 辅助函数模板
 * - 主计算循环模板
 */

// ============================================================
// 类型定义
// ============================================================

export interface ShaderConfig {
    dataType: string;      // f32 或 f16
    storageType: string;   // f32 或 f16
    hasInputC: boolean;    // 是否有 InputC (GEMM)
    hasBatch: boolean;     // 是否为 BMM
}

// ============================================================
// Uniform 结构模板
// ============================================================

/**
 * 生成 MM Uniform 结构（无 batch）
 */
export function generateMmUniformStruct(hasInputC: boolean): string {
    if (hasInputC) {
        return `
struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    offset_a: u32,
    offset_b: u32,
    offset_c: u32,
    offset_out: u32,
    c_shape_m: u32,
    c_shape_n: u32,
    alpha: f32,
    beta: f32,
}
`;
    }
    return `
struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    offset_a: u32,
    offset_b: u32,
    offset_out: u32,
    alpha: f32,
    beta: f32,
}
`;
}

/**
 * 生成 BMM Uniform 结构（有 batch）- 真正的 4D 支持
 * 
 * strides_a/b 是完整的 4D strides (已 padding)
 */
export function generateBmmUniformStruct(hasInputC: boolean): string {
    if (hasInputC) {
        return `
struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    batch_size: u32,
    offset_a: u32,
    offset_b: u32,
    offset_c: u32,
    offset_out: u32,
    ndim_a: u32,
    ndim_b: u32,
    c_shape_m: u32,
    c_shape_n: u32,
    // 完整 4D strides (已 padding 到 4 维)
    strides_a: vec4<u32>,
    strides_b: vec4<u32>,
    alpha: f32,
    beta: f32,
    // --- legacy fields (deprecated) ---
    stride_a_row: u32,
    stride_a_col: u32,
    stride_b_row: u32,
    stride_b_col: u32,
    batch_stride_a: u32,
    batch_stride_b: u32,
}
`;
    }
    return `
struct Uniforms {
    M: u32,
    K: u32,
    N: u32,
    batch_size: u32,
    offset_a: u32,
    offset_b: u32,
    offset_out: u32,
    ndim_a: u32,
    ndim_b: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    // 完整 4D strides (已 padding 到 4 维)
    strides_a: vec4<u32>,
    strides_b: vec4<u32>,
    alpha: f32,
    beta: f32,
    // --- legacy fields (deprecated) ---
    stride_a_row: u32,
    stride_a_col: u32,
    stride_b_row: u32,
    stride_b_col: u32,
    batch_stride_a: u32,
    batch_stride_b: u32,
}
`;
}

// ============================================================
// Buffer 绑定模板
// ============================================================

/**
 * 生成 Buffer 绑定
 */
export function generateBufferBindings(storageType: string, hasInputC: boolean): string {
    const base = `
@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> inputA: array<${storageType}>;
@group(0) @binding(2) var<storage, read> inputB: array<${storageType}>;
@group(0) @binding(3) var<storage, read_write> output: array<${storageType}>;`;

    if (hasInputC) {
        return base + `
@group(0) @binding(4) var<storage, read> inputC: array<${storageType}>;
`;
    }
    return base + '\n';
}

// ============================================================
// Shared Memory 模板
// ============================================================

/**
 * 生成 Shared Memory 声明（标量版本）
 */
export function generateScalarSharedMemory(
    dataType: string,
    tileAOuter: number,
    tileBOuter: number,
    tileInner: number
): string {
    return `
// Shared Memory
var<workgroup> mm_Asub: array<array<${dataType}, ${tileInner}>, ${tileAOuter}>;
var<workgroup> mm_Bsub: array<array<${dataType}, ${tileBOuter}>, ${tileInner}>;
`;
}

/**
 * 生成 Shared Memory 声明（Vec4 版本）
 */
export function generateVec4SharedMemory(
    dataType: string,
    tileAOuter: number,
    tileBOuter: number,
    tileInner: number
): string {
    const vec4Type = `vec4<${dataType}>`;
    return `
// Shared Memory
var<workgroup> mm_Asub: array<array<${vec4Type}, ${tileAOuter / 4}>, ${tileInner}>;
var<workgroup> mm_Bsub: array<array<${vec4Type}, ${tileBOuter / 4}>, ${tileInner}>;
`;
}

// ============================================================
// 辅助函数模板
// ============================================================

/**
 * 生成 mm_readA 函数（MM 版本，使用 uniforms）
 */
export function generateMmReadA(
    dataType: string,
    transposeA: boolean,
    useConstants: boolean = false  // 是否使用常量而非 uniforms
): string {
    const M = useConstants ? 'M_val' : 'uniforms.M';
    const K = useConstants ? 'K_val' : 'uniforms.K';
    const indexExpr = transposeA
        ? `let idx = col * ${M} + row;`
        : `let idx = row * ${K} + col;`;

    return `
fn mm_readA(row: u32, col: u32) -> ${dataType} {
    if (row >= ${M} || col >= ${K}) {
        return ${dataType}(0.0);
    }
    ${indexExpr}
    return ${dataType}(inputA[idx + uniforms.offset_a]);
}
`;
}

/**
 * 生成 mm_readB 函数（MM 版本）
 */
export function generateMmReadB(
    dataType: string,
    transposeB: boolean,
    useConstants: boolean = false
): string {
    const K = useConstants ? 'K_val' : 'uniforms.K';
    const N = useConstants ? 'N_val' : 'uniforms.N';
    const indexExpr = transposeB
        ? `let idx = col * ${K} + row;`
        : `let idx = row * ${N} + col;`;

    return `
fn mm_readB(row: u32, col: u32) -> ${dataType} {
    if (row >= ${K} || col >= ${N}) {
        return ${dataType}(0.0);
    }
    ${indexExpr}
    return ${dataType}(inputB[idx + uniforms.offset_b]);
}
`;
}

/**
 * 生成 mm_readA 函数（BMM 版本，带 batch 参数）
 * 
 * 使用 uniform 中的 strides 计算索引，支持非连续内存布局
 * 索引计算: offset_a + batchA * batch_stride_a + row * stride_a_row + col * stride_a_col
 */
export function generateBmmReadA(
    dataType: string,
    _transposeA: boolean  // 保留参数兼容性，但不再使用 - 完全由 strides 控制
): string {
    return `
fn mm_readA(batch: u32, row: u32, col: u32) -> ${dataType} {
    if (row >= uniforms.M || col >= uniforms.K) {
        return ${dataType}(0.0);
    }
    let batchA = convertBatchIdxA(batch);
    // 使用 strides 计算索引，支持非连续内存布局
    let idx = uniforms.offset_a + batchA * uniforms.batch_stride_a 
            + row * uniforms.stride_a_row + col * uniforms.stride_a_col;
    return ${dataType}(inputA[idx]);
}
`;
}

/**
 * 生成 mm_readB 函数（BMM 版本，带 batch 参数）
 * 
 * 使用 uniform 中的 strides 计算索引，支持非连续内存布局
 * 索引计算: offset_b + batchB * batch_stride_b + row * stride_b_row + col * stride_b_col
 */
export function generateBmmReadB(
    dataType: string,
    _transposeB: boolean  // 保留参数兼容性，但不再使用 - 完全由 strides 控制
): string {
    return `
fn mm_readB(batch: u32, row: u32, col: u32) -> ${dataType} {
    if (row >= uniforms.K || col >= uniforms.N) {
        return ${dataType}(0.0);
    }
    let batchB = convertBatchIdxB(batch);
    // 使用 strides 计算索引，支持非连续内存布局
    let idx = uniforms.offset_b + batchB * uniforms.batch_stride_b 
            + row * uniforms.stride_b_row + col * uniforms.stride_b_col;
    return ${dataType}(inputB[idx]);
}
`;
}

// ============================================================
// 常量定义模板
// ============================================================

/**
 * 生成 tile 常量
 */
export function generateTileConstants(
    rowPerThread: number,
    colPerThread: number,
    tileInner: number,
    M?: number,
    K?: number,
    N?: number
): string {
    let constants = `
// Tile Constants
const rowPerThread = ${rowPerThread}u;
const colPerThread = ${colPerThread}u;
const tileInner = ${tileInner}u;
`;

    // 如果提供了矩阵维度，也添加为常量（用于 Vec4 版本）
    if (M !== undefined && K !== undefined && N !== undefined) {
        constants += `const M_val = ${M}u;
const K_val = ${K}u;
const N_val = ${N}u;
`;
    }

    return constants;
}

/**
 * 生成 BMM 专用常量
 */
export function generateBmmConstants(M: number, K: number, N: number): string {
    return `
// Matrix size constants for batch offset calculation
const MK = ${M * K}u;  // M * K
const KN = ${K * N}u;  // K * N
const MN = ${M * N}u;  // M * N
`;
}

// ============================================================
// BMM 专用模板
// ============================================================

/**
 * 生成 BMM Buffer 绑定
 */
export function generateBmmBufferBindings(storageType: string, hasInputC: boolean): string {
    const base = `
@group(0) @binding(0) var<storage, read> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> inputA: array<${storageType}>;
@group(0) @binding(2) var<storage, read> inputB: array<${storageType}>;
@group(0) @binding(3) var<storage, read_write> output: array<${storageType}>;`;

    if (hasInputC) {
        return base + `
@group(0) @binding(4) var<storage, read> inputC: array<${storageType}>;
`;
    }
    return base + '\n';
}

/**
 * 生成 BMM Shared Memory 声明
 */
export function generateBmmSharedMemory(
    dataType: string,
    tileAOuter: number,
    tileBOuter: number,
    tileInner: number
): string {
    return `
// Shared Memory
var<workgroup> mm_Asub: array<array<${dataType}, ${tileInner}>, ${tileAOuter}>;
var<workgroup> mm_Bsub: array<array<${dataType}, ${tileBOuter}>, ${tileInner}>;
`;
}

/**
 * 生成 BMM Tile 常量（包含矩阵尺寸常量）
 */
export function generateBmmTileConstants(
    rowPerThread: number,
    colPerThread: number,
    tileInner: number,
    M: number,
    K: number,
    N: number
): string {
    return `
// Tile Constants
const rowPerThread = ${rowPerThread}u;
const colPerThread = ${colPerThread}u;
const tileInner = ${tileInner}u;
// Matrix size constants for batch offset calculation
const MK = ${M * K}u;  // M * K
const KN = ${K * N}u;  // K * N
const MN = ${M * N}u;  // M * N
`;
}
