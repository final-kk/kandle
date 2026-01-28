/**
 * MatMul Types - 矩阵乘法算子类型定义
 * 
 * 遵循 GEMM 标准: C = α × A @ B + β × C
 */

import { DType, ITensorHandle, Shape } from '@kandle/types';

// ============================================================
// Tile 配置
// ============================================================

/**
 * Tiling 配置
 * 
 * 根据矩阵尺寸动态选择最优配置
 */
export interface TileConfig {
    /** 是否使用 vec4 向量化加载 */
    useVec4: boolean;

    /** 每个线程计算的元素数 [colPerThread, rowPerThread, batchPerThread] */
    workPerThread: [number, number, number];

    /** 工作组大小 [x, y, z] */
    workgroupSize: [number, number, number];

    /** K 维度分块大小 */
    tileInner: number;
}

// ============================================================
// MatMul 配置
// ============================================================

/**
 * MatMul 内部配置
 * 
 * Dispatcher 构建，传递给 Backend Kernel
 */
export interface MatmulConfig {
    // === 矩阵维度 ===

    /** A 的行数 (M) */
    M: number;

    /** A 的列数 / B 的行数 (K) - 公共维度 */
    K: number;

    /** B 的列数 (N) */
    N: number;

    /** Batch 维度形状 (广播后) */
    batchShape: readonly number[];

    /** Batch 元素总数 */
    batchSize: number;

    // === 转置标志 ===

    /** A 是否转置 */
    transposeA: boolean;

    /** B 是否转置 */
    transposeB: boolean;

    // === GEMM 参数 ===

    /** 乘法缩放因子 (默认 1.0) */
    alpha: number;

    /** 累加缩放因子 (默认 0.0，即不累加) */
    beta: number;

    // === 张量信息 ===

    /** 输入 A - TensorHandle */
    inputA: ITensorHandle;

    /** 输入 B - TensorHandle */
    inputB: ITensorHandle;

    /** 可选的累加/偏置张量 C */
    inputC?: ITensorHandle;

    /** 输出张量 */
    output: ITensorHandle;

    /** 计算类型 */
    computeDtype: DType;

    // === Tile 配置 ===

    /** 动态选择的 Tile 配置 */
    tileConfig: TileConfig;
}

// ============================================================
// 操作变体
// ============================================================

/**
 * MatMul 操作变体
 * 
 * 用于 Dispatcher 路由
 */
export type MatmulVariant =
    | 'dot'   // 1D @ 1D → 标量
    | 'mv'    // 2D @ 1D → 1D 向量
    | 'mm'    // 2D @ 2D → 2D 矩阵
    | 'bmm';  // ≥3D → 带 batch 的矩阵乘法

/**
 * Matmul 分发结果
 * 
 * 由 Dispatcher 构建，传递给 Backend Executor
 */
export interface MatmulDispatchResult {
    /** 操作变体 */
    variant: MatmulVariant;

    /** 输出张量 */
    output: ITensorHandle;

    /** M 维度 (输出行数) */
    M: number;

    /** K 维度 (公共维度) */
    K: number;

    /** N 维度 (输出列数) */
    N: number;

    /** Batch 形状 */
    batchShape: readonly number[];

    /** Batch 大小 */
    batchSize: number;

    /** A 是否转置 (逻辑转置标记，用于 shader 优化路径选择) */
    transposeA: boolean;

    /** B 是否转置 (逻辑转置标记，用于 shader 优化路径选择) */
    transposeB: boolean;

    /** 计算类型 */
    computeDtype: DType;

    // === Strided Memory Support ===

    /**
     * 输入 A 的 strides (最后两维: [strideRow, strideCol])
     * 用于支持非连续内存布局的零拷贝 matmul
     * @deprecated 使用 fullStridesA 代替
     */
    stridesA: readonly number[];

    /**
     * 输入 B 的 strides (最后两维: [strideRow, strideCol])
     * 用于支持非连续内存布局的零拷贝 matmul
     * @deprecated 使用 fullStridesB 代替
     */
    stridesB: readonly number[];

    /**
     * A 每个 batch 的 stride (用于 BMM)
     * 如果是 2D matmul，则为 0
     * @deprecated 使用 fullStridesA 代替
     */
    batchStrideA: number;

    /**
     * B 每个 batch 的 stride (用于 BMM)
     * 如果是 2D matmul，则为 0
     * @deprecated 使用 fullStridesB 代替
     */
    batchStrideB: number;

    // === 真正的 N 维 Strided 支持 (4D BMM) ===

    /**
     * 输入 A 的完整 strides (所有维度)
     * 例如 4D tensor [B, H, M, K] 的 strides: [H*M*K, M*K, K, 1] 或任意非连续模式
     * padding 到 4 维: 如果 ndim < 4，则前面补 0
     */
    fullStridesA: readonly number[];

    /**
     * 输入 B 的完整 strides (所有维度)
     * 例如 4D tensor [B, H, K, N] 的 strides
     * padding 到 4 维: 如果 ndim < 4，则前面补 0
     */
    fullStridesB: readonly number[];

    /**
     * 输入 A 的实际维度数
     */
    ndimA: number;

    /**
     * 输入 B 的实际维度数
     */
    ndimB: number;

    // === GEMM 参数 (Phase 4) ===

    /** 乘法缩放因子 (默认 1.0) */
    alpha: number;

    /** 累加缩放因子 (默认 0.0) */
    beta: number;

    /** 可选的累加/偏置张量 C (用于 addmm, baddbmm 等) */
    inputC?: ITensorHandle;
}

/**
 * 检查输入是否为 matmul 支持的类型
 * 
 * 支持的类型:
 * - float16, float32, float64 (实数浮点)
 * - complex64, complex128 (复数)
 * 
 * 遵循 PyTorch 语义: matmul 只支持浮点和复数类型
 */
export function isValidMatmulDType(dtype: DType): boolean {
    return (
        dtype === 'float32' ||
        dtype === 'float16' ||
        dtype === 'float64' ||
        dtype === 'complex64' ||
        dtype === 'complex128'
    );
}

/**
 * 判断张量是否是转置视图
 * 
 * 通过检查 stride 判断：如果最后一个 stride 不是 1，或者倒数第二个 stride 是 1，则可能是转置
 * 
 * @param shape 张量形状
 * @param strides 张量步长
 * @returns 是否是转置视图
 */
export function isTransposed(shape: readonly number[], strides: readonly number[]): boolean {
    if (shape.length < 2) return false;

    const rank = shape.length;
    const lastStride = strides[rank - 1];
    const secondLastStride = strides[rank - 2];

    // 正常行主序: strides = [..., N, 1]
    // 转置后: strides = [..., 1, M] (假设原来是 [M, N])
    // 判断: 如果 lastStride > secondLastStride，则是转置
    return lastStride > secondLastStride && secondLastStride === 1;
}

/**
 * 检查矩阵最后两维是否内存连续
 */
export function isContiguousLast2Dims(
    shape: readonly number[],
    strides: readonly number[]
): boolean {
    if (shape.length < 2) return true;

    const rank = shape.length;
    const expectedStride = shape[rank - 1]; // 最后一维的大小

    // 检查最后一维 stride 是否为 1
    if (strides[rank - 1] !== 1) return false;

    // 检查倒数第二维 stride 是否等于最后一维大小
    if (strides[rank - 2] !== expectedStride) return false;

    return true;
}

// ============================================================
// Shader Cache Key
// ============================================================

/**
 * GEMM 变体类型（用于缓存键）
 */
export type GemmVariant = 'beta0' | 'alpha1-beta1' | 'general';

/**
 * 计算 GEMM 变体
 */
export function computeGemmVariant(alpha: number, beta: number, hasInputC: boolean): GemmVariant {
    const isPureMatmul = !hasInputC || beta === 0.0;
    const isSimpleAddmm = hasInputC && alpha === 1.0 && beta === 1.0;

    if (isPureMatmul) return 'beta0';
    if (isSimpleAddmm) return 'alpha1-beta1';
    return 'general';
}

/**
 * Matmul 缓存键配置接口
 * 
 * 包含所有影响 shader 生成的参数，用于高效缓存而不需要重新生成 shader
 */
export interface MatmulCacheKey {
    /** 操作变体 */
    variant: MatmulVariant;
    /** 计算数据类型 */
    dtype: DType;
    /** M 维度 */
    M: number;
    /** K 维度 */
    K: number;
    /** N 维度 */
    N: number;
    /** A 是否转置 */
    transposeA: boolean;
    /** B 是否转置 */
    transposeB: boolean;
    /** 是否使用 vec4 */
    useVec4: boolean;
    /** GEMM 变体 */
    gemmVariant: GemmVariant;
    /** batch 形状 A (仅 BMM) */
    batchShapeA?: string;
    /** batch 形状 B (仅 BMM) */
    batchShapeB?: string;
    /** batch 形状 C (仅 BMM with GEMM) */
    batchShapeC?: string;
}

/**
 * 计算 MM 的 pipeline cache key
 */
export function computeMmCacheKey(
    dtype: DType,
    M: number,
    K: number,
    N: number,
    transposeA: boolean,
    transposeB: boolean,
    useVec4: boolean,
    gemmVariant: GemmVariant
): string {
    return `matmul.mm-${dtype}-${M}-${K}-${N}-${transposeA}-${transposeB}-${useVec4}-${gemmVariant}`;
}

/**
 * 计算 BMM 的 pipeline cache key
 */
export function computeBmmCacheKey(
    dtype: DType,
    M: number,
    K: number,
    N: number,
    batchSize: number,
    transposeA: boolean,
    transposeB: boolean,
    batchShapeA: readonly number[],
    batchShapeB: readonly number[],
    batchShapeC: readonly number[] | undefined,
    gemmVariant: GemmVariant
): string {
    const batchShapeCKey = batchShapeC ? `C${batchShapeC.join(',')}` : 'C-';
    return `matmul.bmm-${dtype}-${M}-${K}-${N}-${batchSize}-${transposeA}-${transposeB}-A${batchShapeA.join(',')}-B${batchShapeB.join(',')}-${batchShapeCKey}-${gemmVariant}`;
}

/**
 * 计算专用 Dot kernel 的 cache key
 */
export function computeDotCacheKey(dtype: DType, K: number): string {
    return `matmul.dot-${dtype}-${K}`;
}

/**
 * 计算专用 MV kernel 的 cache key
 */
export function computeMvCacheKey(
    dtype: DType,
    M: number,
    K: number,
    transposeA: boolean
): string {
    return `matmul.mv-${dtype}-${M}-${K}-${transposeA}`;
}

