/**
 * MatMul Operations Dispatcher
 * 
 * 矩阵乘法分发器 - 不走 TensorIterator，使用专用路径
 * 
 * 职责:
 * 1. 维度路由 (1D@1D→dot, 2D@1D→mv, 2D@2D→mm, ≥3D→bmm)
 * 2. 形状验证和输出形状计算
 * 3. Batch 广播形状计算
 * 4. 类型检查和提升
 * 5. 构建 MatmulConfig 传递给 Backend
 */

import { ITensorHandle, DType, Shape } from '@kandle/types';
import { Logger, computeBroadcastShape } from '@kandle/utils';
import { env } from '../env';

const logger = new Logger('Dispatch-Matmul');

// ============================================================
// 类型定义
// ============================================================

/**
 * MatMul 操作变体
 */
export type MatmulVariant = 'dot' | 'mv' | 'mm' | 'bmm';

/**
 * Matmul 分发结果
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

    // === Strided Memory Support (Legacy) ===

    /**
     * 输入 A 的 strides (最后两维: [strideRow, strideCol])
     * @deprecated 使用 fullStridesA 代替
     */
    stridesA: readonly number[];

    /**
     * 输入 B 的 strides (最后两维: [strideRow, strideCol])
     * @deprecated 使用 fullStridesB 代替
     */
    stridesB: readonly number[];

    /**
     * A 每个 batch 的 stride (用于 BMM)
     * @deprecated 使用 fullStridesA 代替
     */
    batchStrideA: number;

    /**
     * B 每个 batch 的 stride (用于 BMM)
     * @deprecated 使用 fullStridesB 代替
     */
    batchStrideB: number;

    // === 真正的 N 维 Strided 支持 (4D BMM) ===

    /**
     * 输入 A 的完整 strides (所有维度)
     * 例如 4D tensor [B, H, M, K] 的 strides
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


// ============================================================
// 辅助函数
// ============================================================

/**
 * 检查是否为有效的 matmul dtype
 * 支持: float16, float32, float64, complex64, complex128
 */
function isValidMatmulDType(dtype: DType): boolean {
    return (
        dtype === 'float32' ||
        dtype === 'float16' ||
        dtype === 'float64' ||
        dtype === 'complex64' ||
        dtype === 'complex128'
    );
}

/**
 * 计算 common dtype
 * 复数类型优先于实数类型
 */
function promoteMatmulDType(dtypeA: DType, dtypeB: DType): DType {
    // 复数优先级: complex128 > complex64
    if (dtypeA === 'complex128' || dtypeB === 'complex128') return 'complex128';
    if (dtypeA === 'complex64' || dtypeB === 'complex64') return 'complex64';
    // 实数优先级: float64 > float32 > float16
    if (dtypeA === 'float64' || dtypeB === 'float64') return 'float64';
    if (dtypeA === 'float32' || dtypeB === 'float32') return 'float32';
    return 'float16';
}

/**
 * 检测张量是否为转置视图
 * 
 * R8: 增强的转置检测
 * 
 * 转置视图的特征：
 * - 最后一维 stride > 倒数第二维 stride
 * - 倒数第二维 stride == 1 (列主序)
 * 
 * 零拷贝支持：通过检测 stride 模式，避免在 matmul 前创建连续副本
 */
function detectTranspose(handle: ITensorHandle): boolean {
    const { shape, strides } = handle;
    if (shape.length < 2) return false;

    const rank = shape.length;
    const lastStride = strides[rank - 1];
    const secondLastStride = strides[rank - 2];

    // 转置: 最后一维 stride > 倒数第二维 stride，且倒数第二维 stride 为 1
    return lastStride > secondLastStride && secondLastStride === 1;
}

/**
 * R8: 检测矩阵是否为标准行主序（连续存储）
 * 
 * 行主序特征：
 * - 最后一维 stride == 1
 * - 倒数第二维 stride == 最后一维大小
 */
function isRowMajor(handle: ITensorHandle): boolean {
    const { shape, strides } = handle;
    if (shape.length < 2) return true;  // 1D 始终连续

    const rank = shape.length;
    const lastStride = strides[rank - 1];
    const secondLastStride = strides[rank - 2];
    const lastDimSize = shape[rank - 1];

    return lastStride === 1 && secondLastStride === lastDimSize;
}

/**
 * R8: 检测是否需要连续化（非转置且非行主序）
 * 
 * 如果张量既不是行主序也不是简单转置，
 * 可能需要在 matmul 前创建连续副本
 * （当前实现仅支持行主序和转置视图）
 */
function needsContiguous(handle: ITensorHandle): boolean {
    const { shape, strides } = handle;
    if (shape.length < 2) return false;

    const isContiguous = isRowMajor(handle);
    const isTransposed = detectTranspose(handle);

    // 如果既不连续也不是简单转置，可能需要额外处理
    // 当前 shader 支持这两种情况的零拷贝
    return !isContiguous && !isTransposed;
}

/**
 * 获取物理矩阵的 strides [strideRow, strideCol]
 * 
 * 返回物理存储的行/列 strides，用于 shader 的索引计算。
 * 
 * **关键**：transpose 参数只是一个标记，告诉 shader 如何解释逻辑坐标。
 * 这里返回的是**物理** strides，不需要根据 transpose 交换。
 * Shader 会根据 transposeB 标志自行选择正确的索引公式：
 *   - transposeB=false: idx = row * stride_row + col * stride_col
 *   - transposeB=true:  idx = col * stride_row + row * stride_col
 * 
 * @param handle tensor 句柄
 * @param _transpose 未使用，保留参数兼容性
 */
function getMatrixStrides(handle: ITensorHandle, _transpose: boolean): readonly [number, number] {
    const { strides, shape } = handle;
    const rank = shape.length;

    if (rank === 0) {
        return [1, 1];
    }
    if (rank === 1) {
        // 1D 向量：作为行向量 [1, K] 处理
        // logicalRow=0 固定，logicalCol 使用 strides[0]
        return [1, strides[0]];
    }

    // 返回物理矩阵的 strides，不考虑逻辑转置
    // strideRow = strides[rank-2] (沿着物理行移动)
    // strideCol = strides[rank-1] (沿着物理列移动)
    return [strides[rank - 2], strides[rank - 1]];
}

/**
 * 计算 batch stride (用于 BMM)
 * 
 * 对于形状 [...batch, M, K] 的 tensor，batch stride 是 M*K 位置之间的距离
 */
function getBatchStride(handle: ITensorHandle): number {
    const { strides, shape } = handle;
    const rank = shape.length;

    if (rank <= 2) {
        return 0;  // 没有 batch 维度
    }

    // batch stride 是倒数第三维的 stride
    return strides[rank - 3];
}

/**
 * 将 strides padding 到 4 维
 * 
 * 用于真正的 4D BMM 支持。如果 ndim < 4，在前面补 0。
 * 
 * 例如:
 * - 2D [M, K] strides [K, 1] -> [0, 0, K, 1]
 * - 3D [B, M, K] strides [M*K, K, 1] -> [0, M*K, K, 1]
 * - 4D [B, H, M, K] strides [H*M*K, M*K, K, 1] -> [H*M*K, M*K, K, 1]
 */
function padStridesTo4D(strides: readonly number[]): readonly [number, number, number, number] {
    const len = strides.length;
    if (len === 0) {
        return [0, 0, 0, 1];
    }
    if (len === 1) {
        return [0, 0, 0, strides[0]];
    }
    if (len === 2) {
        return [0, 0, strides[0], strides[1]];
    }
    if (len === 3) {
        return [0, strides[0], strides[1], strides[2]];
    }
    // len >= 4: 取最后 4 个维度
    return [
        strides[len - 4],
        strides[len - 3],
        strides[len - 2],
        strides[len - 1]
    ];
}

/**
 * 获取矩阵最后两维的逻辑形状 (考虑转置)
 * 
 * R8: 零拷贝支持 - 根据转置标志返回逻辑维度
 * 
 * @param shape 物理形状
 * @param isTransposed 是否逻辑转置
 * @returns [rows, cols] 逻辑维度
 */
function getMatrixDims(shape: readonly number[], isTransposed: boolean): [number, number] {
    const rank = shape.length;
    if (rank < 2) {
        throw new Error(`Expected at least 2D tensor, got ${rank}D`);
    }

    const rows = shape[rank - 2];
    const cols = shape[rank - 1];

    // R8: 转置时交换逻辑维度，但物理存储不变
    return isTransposed ? [cols, rows] : [rows, cols];
}

/**
 * 计算输出形状
 */
function computeOutputShape(
    batchShape: readonly number[],
    M: number,
    N: number
): number[] {
    return [...batchShape, M, N];
}

/**
 * 验证 inputC 形状是否可以广播到输出形状
 * 
 * 支持的 C 形状:
 * - 标量 [] - 广播到所有位置
 * - 行向量 [1, N] 或 [N] - 广播到每一行
 * - 列向量 [M, 1] - 广播到每一列  
 * - 完整矩阵 [M, N] - 逐元素相加
 * - Batch 矩阵 [...batch, M, N] - 支持 batch 广播
 * 
 * @param outputShape 输出张量形状
 * @param cShape inputC 形状
 * @throws 如果形状不兼容
 */
function validateInputCShape(outputShape: readonly number[], cShape: readonly number[]): void {
    // 标量广播
    if (cShape.length === 0) {
        return; // 标量可以广播到任意形状
    }

    // 检查是否可以广播：从右边对齐
    const outputRank = outputShape.length;
    const cRank = cShape.length;

    // C 的秩不能大于输出秩
    if (cRank > outputRank) {
        throw new Error(
            `inputC shape ${JSON.stringify(cShape)} cannot be broadcast to output shape ${JSON.stringify(outputShape)} ` +
            `(C has more dimensions)`
        );
    }

    // 从右边对齐，检查每一维是否可以广播
    for (let i = 0; i < cRank; i++) {
        const cDim = cShape[cRank - 1 - i];
        const outDim = outputShape[outputRank - 1 - i];

        // 广播规则：维度必须相等或其中一个为 1
        if (cDim !== outDim && cDim !== 1) {
            throw new Error(
                `inputC shape ${JSON.stringify(cShape)} cannot be broadcast to output shape ${JSON.stringify(outputShape)} ` +
                `(dimension ${cRank - 1 - i}: ${cDim} vs ${outDim})`
            );
        }
    }
}

// ============================================================
// 主分发函数
// ============================================================

/**
 * 分发 matmul 操作
 * 
 * 遵循 PyTorch torch.matmul 语义:
 * - 1D @ 1D: dot product → scalar
 * - 2D @ 1D: matrix-vector → 1D
 * - 1D @ 2D: vector-matrix → 1D  
 * - 2D @ 2D: matrix-matrix → 2D
 * - ≥3D: batched matmul with broadcasting
 * 
 * 支持完整的 GEMM: out = beta * C + alpha * (A @ B)
 * 
 * @param a 输入张量 A
 * @param b 输入张量 B
 * @param c 可选的累加张量 C (用于 addmm 等)
 * @param alpha 乘法缩放因子 (默认 1.0)
 * @param beta 累加缩放因子 (默认 0.0)
 * @param out 可选的输出张量
 * @returns 输出张量
 */
export function dispatchMatmul(
    a: ITensorHandle,
    b: ITensorHandle,
    c?: ITensorHandle,
    alpha: number = 1.0,
    beta: number = 0.0,
    out?: ITensorHandle,
    transposeA?: boolean,
    transposeB?: boolean
): ITensorHandle {
    logger.debug(`Dispatching matmul: A${JSON.stringify(a.shape)} @ B${JSON.stringify(b.shape)}, alpha=${alpha}, beta=${beta}`);

    // 1. 设备检查
    if (a.device !== b.device) {
        throw new Error(`matmul requires tensors on same device, got ${a.device} and ${b.device}`);
    }
    if (c && c.device !== a.device) {
        throw new Error(`matmul: inputC must be on same device as A and B`);
    }

    // 2. 类型检查
    if (!isValidMatmulDType(a.dtype)) {
        throw new Error(`matmul does not support dtype ${a.dtype}, use float16/32/64`);
    }
    if (!isValidMatmulDType(b.dtype)) {
        throw new Error(`matmul does not support dtype ${b.dtype}, use float16/32/64`);
    }
    if (c && !isValidMatmulDType(c.dtype)) {
        throw new Error(`matmul: inputC does not support dtype ${c.dtype}, use float16/32/64`);
    }

    // 3. Beta/C 一致性检查
    if (beta !== 0.0 && !c) {
        throw new Error(`matmul: beta=${beta} requires inputC to be provided`);
    }
    if (c && beta === 0.0) {
        logger.warn('matmul: inputC provided but beta=0, C will be ignored');
    }

    const dimA = a.shape.length;
    const dimB = b.shape.length;

    // 4. 维度路由
    let result: MatmulDispatchResult;

    if (dimA === 1 && dimB === 1) {
        // Case 1: 向量点积 1D @ 1D → scalar
        result = dispatchDot(a, b, out);
    } else if (dimA === 2 && dimB === 1) {
        // Case 2: 矩阵向量乘 2D @ 1D → 1D
        result = dispatchMv(a, b, out);
    } else if (dimA === 1 && dimB === 2) {
        // Case 3: 向量矩阵乘 1D @ 2D → 1D
        // 等价于 unsqueeze(0) → mm → squeeze(0)
        result = dispatchVm(a, b, out);
    } else if (dimA === 2 && dimB === 2) {
        // Case 4: 标准矩阵乘 2D @ 2D → 2D
        result = dispatchMm(a, b, out, transposeA, transposeB);
    } else if (dimA >= 1 && dimB >= 1) {
        // Case 5: 批次矩阵乘（带广播）
        result = dispatchBmm(a, b, out, transposeA, transposeB);
    } else {
        throw new Error(`Invalid shapes for matmul: ${JSON.stringify(a.shape)} and ${JSON.stringify(b.shape)}`);
    }

    // 5. 设置 GEMM 参数
    result.alpha = alpha;
    result.beta = beta;
    result.inputC = c;

    // 6. 如果有 inputC，验证形状兼容性
    if (c && beta !== 0.0) {
        validateInputCShape(result.output.shape, c.shape);
    }

    logger.debug(`Variant: ${result.variant}, M=${result.M}, K=${result.K}, N=${result.N}, batch=${JSON.stringify(result.batchShape)}`);

    // 7. 调用 Backend Kernel
    const backend = env.getBackend(a.device);
    const kernel = backend.operators.find('matmul') as unknown as (config: MatmulDispatchResult, a: ITensorHandle, b: ITensorHandle) => void;

    if (!kernel) {
        throw new Error(`matmul kernel not registered for device ${a.device}`);
    }

    kernel(result, a, b);

    return result.output;
}

// ============================================================
// 变体分发
// ============================================================

/**
 * Case 1: Dot product (1D @ 1D → scalar)
 */
function dispatchDot(
    a: ITensorHandle,
    b: ITensorHandle,
    out?: ITensorHandle
): MatmulDispatchResult {
    const K = a.shape[0];

    if (b.shape[0] !== K) {
        throw new Error(`dot: shapes ${a.shape} and ${b.shape} not aligned`);
    }

    const backend = env.getBackend(a.device);
    const resultDtype = promoteMatmulDType(a.dtype, b.dtype);

    // 输出是标量
    const outputShape: Shape = [];
    const output = out ?? backend.createTensorHandle(outputShape, resultDtype);

    return {
        variant: 'dot',
        output,
        M: 1,
        K,
        N: 1,
        batchShape: [],
        batchSize: 1,
        transposeA: false,
        transposeB: false,
        computeDtype: resultDtype,
        stridesA: [a.strides[0], 1],
        stridesB: [b.strides[0], 1],
        batchStrideA: 0,
        batchStrideB: 0,
        fullStridesA: padStridesTo4D(a.strides),
        fullStridesB: padStridesTo4D(b.strides),
        ndimA: a.shape.length,
        ndimB: b.shape.length,
        alpha: 1.0,
        beta: 0.0,
    };
}

/**
 * Case 2: Matrix-Vector (2D @ 1D → 1D)
 */
function dispatchMv(
    a: ITensorHandle,
    b: ITensorHandle,
    out?: ITensorHandle
): MatmulDispatchResult {
    const [M, K] = a.shape as [number, number];
    const K_b = b.shape[0];

    if (K !== K_b) {
        throw new Error(`mv: shapes [${M}, ${K}] and [${K_b}] not aligned (K mismatch)`);
    }

    const backend = env.getBackend(a.device);
    const resultDtype = promoteMatmulDType(a.dtype, b.dtype);

    // 输出是向量

    const outputShape: Shape = [M];
    const output = out ?? backend.createTensorHandle(outputShape, resultDtype);

    return {
        variant: 'mv',
        output,
        M,
        K,
        N: 1,
        batchShape: [],
        batchSize: 1,
        transposeA: detectTranspose(a),
        transposeB: false,
        computeDtype: resultDtype,
        stridesA: getMatrixStrides(a, detectTranspose(a)),
        stridesB: [b.strides[0], 1],
        batchStrideA: 0,
        batchStrideB: 0,
        fullStridesA: padStridesTo4D(a.strides),
        fullStridesB: padStridesTo4D(b.strides),
        ndimA: a.shape.length,
        ndimB: b.shape.length,
        alpha: 1.0,
        beta: 0.0,
    };
}

/**
 * Case 3: Vector-Matrix (1D @ 2D → 1D)
 * 等价于 A.unsqueeze(0) @ B → squeeze(0)
 * 
 * 使用 MM kernel 而非 MV kernel，因为：
 * - VM: [K] @ [K, N] = [1, K] @ [K, N] = [1, N] -> squeeze -> [N]
 * - 这是 [1, K] 矩阵乘以 [K, N] 矩阵，应该使用 MM
 * - MV 专用 kernel 期望的是 [M, K] @ [K] 形式
 */
function dispatchVm(
    a: ITensorHandle,
    b: ITensorHandle,
    out?: ITensorHandle
): MatmulDispatchResult {
    const K = a.shape[0];
    const [K_b, N] = b.shape as [number, number];

    if (K !== K_b) {
        throw new Error(`vm: shapes [${K}] and [${K_b}, ${N}] not aligned (K mismatch)`);
    }

    const backend = env.getBackend(a.device);
    const resultDtype = promoteMatmulDType(a.dtype, b.dtype);

    // 输出是向量
    const outputShape: Shape = [N];
    const output = out ?? backend.createTensorHandle(outputShape, resultDtype);

    return {
        variant: 'mm', // 使用 MM kernel: [1, K] @ [K, N] = [1, N]
        output,
        M: 1,
        K,
        N,
        batchShape: [],
        batchSize: 1,
        transposeA: false, // 1D 张量没有转置
        transposeB: detectTranspose(b),
        computeDtype: resultDtype,
        stridesA: [a.strides[0], 1],
        stridesB: getMatrixStrides(b, detectTranspose(b)),
        batchStrideA: 0,
        batchStrideB: 0,
        fullStridesA: padStridesTo4D(a.strides),
        fullStridesB: padStridesTo4D(b.strides),
        ndimA: a.shape.length,
        ndimB: b.shape.length,
        alpha: 1.0,
        beta: 0.0,
    };
}

/**
 * Case 4: Matrix-Matrix (2D @ 2D → 2D)
 */
function dispatchMm(
    a: ITensorHandle,
    b: ITensorHandle,
    out?: ITensorHandle,
    userTransposeA?: boolean,
    userTransposeB?: boolean
): MatmulDispatchResult {
    // User-specified transpose overrides auto-detection
    const transposeA = userTransposeA ?? detectTranspose(a);
    const transposeB = userTransposeB ?? detectTranspose(b);

    // 获取逻辑维度（考虑转置）
    const [M, K_a] = getMatrixDims(a.shape, transposeA);
    const [K_b, N] = getMatrixDims(b.shape, transposeB);

    if (K_a !== K_b) {
        throw new Error(`mm: shapes ${JSON.stringify(a.shape)} and ${JSON.stringify(b.shape)} not aligned (K: ${K_a} vs ${K_b})`);
    }

    const backend = env.getBackend(a.device);
    const resultDtype = promoteMatmulDType(a.dtype, b.dtype);

    const outputShape: Shape = [M, N];
    const output = out ?? backend.createTensorHandle(outputShape, resultDtype);

    return {
        variant: 'mm',
        output,
        M,
        K: K_a,
        N,
        batchShape: [],
        batchSize: 1,
        transposeA,
        transposeB,
        computeDtype: resultDtype,
        stridesA: getMatrixStrides(a, transposeA),
        stridesB: getMatrixStrides(b, transposeB),
        batchStrideA: 0,
        batchStrideB: 0,
        fullStridesA: padStridesTo4D(a.strides),
        fullStridesB: padStridesTo4D(b.strides),
        ndimA: a.shape.length,
        ndimB: b.shape.length,
        alpha: 1.0,
        beta: 0.0,
    };
}

/**
 * Case 5: Batched Matrix-Matrix (≥3D @ ≥3D → ≥3D, with broadcasting)
 */
function dispatchBmm(
    a: ITensorHandle,
    b: ITensorHandle,
    out?: ITensorHandle,
    userTransposeA?: boolean,
    userTransposeB?: boolean
): MatmulDispatchResult {
    const dimA = a.shape.length;
    const dimB = b.shape.length;

    // 分离 batch 维度和矩阵维度
    // A: [...batchA, M, K]
    // B: [...batchB, K, N]

    // 如果是 1D，先视为 2D
    let shapeA = a.shape;
    let shapeB = b.shape;
    let squeezeFront = false;
    let squeezeBack = false;

    if (dimA === 1) {
        // 1D @ ND: 视为 [1, K] @ [..., K, N]
        shapeA = [1, ...a.shape];
        squeezeFront = true;
    }

    if (dimB === 1) {
        // ND @ 1D: 视为 [..., M, K] @ [K, 1]
        shapeB = [...b.shape, 1];
        squeezeBack = true;
    }

    const rankA = shapeA.length;
    const rankB = shapeB.length;

    // User-specified transpose overrides auto-detection
    const transposeA = userTransposeA ?? (dimA > 1 && detectTranspose(a));
    const transposeB = userTransposeB ?? (dimB > 1 && detectTranspose(b));

    // 获取矩阵维度
    const M = shapeA[rankA - 2];
    const K_a = shapeA[rankA - 1];
    const K_b = shapeB[rankB - 2];
    const N = shapeB[rankB - 1];

    // 考虑转置调整 K
    const effectiveK_a = transposeA ? M : K_a;
    const effectiveK_b = transposeB ? N : K_b;
    const effectiveM = transposeA ? K_a : M;
    const effectiveN = transposeB ? K_b : N;

    if (effectiveK_a !== effectiveK_b) {
        throw new Error(
            `bmm: shapes ${JSON.stringify(a.shape)} and ${JSON.stringify(b.shape)} ` +
            `not aligned (K: ${effectiveK_a} vs ${effectiveK_b})`
        );
    }

    // 计算 batch 广播形状
    const batchA = shapeA.slice(0, -2);
    const batchB = shapeB.slice(0, -2);

    let batchShape: readonly number[];
    try {
        batchShape = computeBroadcastShape(batchA, batchB);
    } catch (e) {
        throw new Error(
            `bmm: batch dimensions not broadcastable: ${JSON.stringify(batchA)} and ${JSON.stringify(batchB)}`
        );
    }

    const batchSize = batchShape.reduce((acc, dim) => acc * dim, 1);

    const backend = env.getBackend(a.device);
    const resultDtype = promoteMatmulDType(a.dtype, b.dtype);

    // 计算输出形状
    let outputShape: Shape = [...batchShape, effectiveM, effectiveN];

    // 恢复 squeeze 操作
    if (squeezeFront) {
        // 移除结果的倒数第二维 (M=1)
        outputShape = outputShape.filter((_, i) => i !== outputShape.length - 2);
    }
    if (squeezeBack) {
        // 移除结果的最后一维 (N=1)
        outputShape = outputShape.slice(0, -1);
    }

    const output = out ?? backend.createTensorHandle(outputShape, resultDtype);

    return {
        variant: 'bmm',
        output,
        M: effectiveM,
        K: effectiveK_a,
        N: effectiveN,
        batchShape,
        batchSize,
        transposeA,
        transposeB,
        computeDtype: resultDtype,
        stridesA: getMatrixStrides(a, transposeA),
        stridesB: getMatrixStrides(b, transposeB),
        batchStrideA: getBatchStride(a),
        batchStrideB: getBatchStride(b),
        fullStridesA: padStridesTo4D(a.strides),
        fullStridesB: padStridesTo4D(b.strides),
        ndimA: a.shape.length,
        ndimB: b.shape.length,
        alpha: 1.0,
        beta: 0.0,
    };
}

// ============================================================
// 便捷 API
// ============================================================

/**
 * mm - 严格 2D 矩阵乘法
 */
export function dispatchMmStrict(
    a: ITensorHandle,
    b: ITensorHandle,
    out?: ITensorHandle
): ITensorHandle {
    if (a.shape.length !== 2 || b.shape.length !== 2) {
        throw new Error(`mm requires 2D tensors, got ${a.shape.length}D and ${b.shape.length}D`);
    }
    return dispatchMatmul(a, b, out);
}

/**
 * bmm - 严格 3D 批次矩阵乘法
 */
export function dispatchBmmStrict(
    a: ITensorHandle,
    b: ITensorHandle,
    out?: ITensorHandle
): ITensorHandle {
    if (a.shape.length !== 3 || b.shape.length !== 3) {
        throw new Error(`bmm requires 3D tensors, got ${a.shape.length}D and ${b.shape.length}D`);
    }
    if (a.shape[0] !== b.shape[0]) {
        throw new Error(`bmm requires same batch size, got ${a.shape[0]} and ${b.shape[0]}`);
    }
    return dispatchMatmul(a, b, out);
}
