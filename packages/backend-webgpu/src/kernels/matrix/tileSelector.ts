/**
 * Tile Selector - 动态选择最优 Tiling 配置
 * 
 * R5: Tile 配置自动调优
 * 参考 ONNX Runtime WebGPU 实现的启发式规则，结合 GPU 设备特性
 */

import { DType } from '@kandle/types';
import { TileConfig } from './types';
import { WebGPUDeviceManager } from '../../base/device';

// ============================================================
// GPU Limits 接口 (用于依赖注入，便于测试)
// ============================================================

export interface GPULimits {
    maxComputeWorkgroupSizeX: number;
    maxComputeWorkgroupSizeY: number;
    maxComputeWorkgroupSizeZ: number;
    maxComputeInvocationsPerWorkgroup: number;
    maxComputeWorkgroupStorageSize: number;
    maxComputeWorkgroupsPerDimension: number;
}

/**
 * 获取当前 GPU 的 limits
 */
function getGPULimits(): GPULimits {
    const limits = WebGPUDeviceManager.limits;
    return {
        maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
        maxComputeWorkgroupSizeY: limits.maxComputeWorkgroupSizeY,
        maxComputeWorkgroupSizeZ: limits.maxComputeWorkgroupSizeZ,
        maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize,
        maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
    };
}

// ============================================================
// 配置策略
// ============================================================

/**
 * 矩阵分类
 * 
 * 用于启发式规则选择
 */
type MatrixCategory =
    | 'tiny'      // M*N < 64，非常小的矩阵
    | 'skinny'    // M 很小 (M <= 8)，典型的矮矩阵
    | 'tall'      // N 很小 (N <= 8)，典型的高矩阵
    | 'small'     // M*N < 4096，小矩阵
    | 'medium'    // M*N < 65536，中等矩阵
    | 'large';    // M*N >= 65536，大矩阵

function classifyMatrix(M: number, N: number, K: number): MatrixCategory {
    const outputSize = M * N;

    if (outputSize < 64) return 'tiny';
    if (M <= 8) return 'skinny';
    if (N <= 8) return 'tall';
    if (outputSize < 4096) return 'small';
    if (outputSize < 65536) return 'medium';
    return 'large';
}

/**
 * 获取数据类型的字节大小
 */
function getBytesPerElement(dtype: DType): number {
    switch (dtype) {
        case 'float16': return 2;
        case 'float32': return 4;
        case 'float64': return 8;
        default: return 4; // 默认 f32
    }
}

/**
 * 验证 tile 配置是否在 GPU 限制范围内
 */
function isValidTileConfig(
    config: TileConfig,
    bytesPerElement: number,
    limits: GPULimits
): boolean {
    const [wgX, wgY, wgZ] = config.workgroupSize;
    const [colPerThread, rowPerThread, batchPerThread] = config.workPerThread;
    const { tileInner, useVec4 } = config;

    // 检查工作组大小限制
    if (wgX > limits.maxComputeWorkgroupSizeX ||
        wgY > limits.maxComputeWorkgroupSizeY ||
        wgZ > limits.maxComputeWorkgroupSizeZ) {
        return false;
    }

    // 检查总 invocations 限制
    const totalInvocations = wgX * wgY * wgZ;
    if (totalInvocations > limits.maxComputeInvocationsPerWorkgroup) {
        return false;
    }

    // 检查 shared memory 限制
    const sharedMemBytes = calculateSharedMemoryBytes(config, bytesPerElement);
    if (sharedMemBytes > limits.maxComputeWorkgroupStorageSize) {
        return false;
    }

    return true;
}

/**
 * 根据矩阵尺寸和数据类型选择最优 Tile 配置
 * 
 * R5: 自动调优
 * 
 * 启发式规则:
 * 1. 检查 GPU 设备限制，确保配置有效
 * 2. 根据矩阵类别选择不同策略
 * 3. 如果 K 和 N 能被 4 整除，优先使用 vec4 向量化加载
 * 4. 小矩阵使用较小的 workgroup，大矩阵使用较大的 tile
 * 5. FP16 使用更激进的 tiling（内存带宽更小）
 * 
 * @param M 矩阵 A 的行数
 * @param K 公共维度
 * @param N 矩阵 B 的列数
 * @param dtype 数据类型
 * @param customLimits 可选的自定义 GPU 限制（用于测试）
 */
export function selectTileConfig(
    M: number,
    K: number,
    N: number,
    dtype: DType,
    customLimits?: GPULimits
): TileConfig {
    const limits = customLimits ?? getGPULimits();
    const category = classifyMatrix(M, N, K);
    const bytesPerElement = getBytesPerElement(dtype);

    // 检查是否可以使用 vec4 优化
    // vec4 要求 K 和 N 都是 4 的倍数
    const canUseVec4 = K % 4 === 0 && N % 4 === 0;

    // 根据矩阵类别选择配置策略
    let candidates: TileConfig[] = [];

    switch (category) {
        case 'tiny':
            // 非常小的矩阵，使用最小配置
            candidates = [
                {
                    useVec4: false,
                    workPerThread: [1, 1, 1],
                    workgroupSize: [4, 4, 1],
                    tileInner: 8,
                },
                {
                    useVec4: false,
                    workPerThread: [1, 1, 1],
                    workgroupSize: [8, 8, 1],
                    tileInner: 16,
                },
            ];
            break;

        case 'skinny':
            // 矮矩阵特化 (M <= 8)
            // 每个线程只处理少量行，但处理多列
            if (canUseVec4) {
                candidates = [
                    {
                        useVec4: true,
                        workPerThread: [4, 1, 1],  // 4 列，1 行
                        workgroupSize: [8, 8, 1],
                        tileInner: 32,
                    },
                    {
                        useVec4: true,
                        workPerThread: [4, 2, 1],  // 4 列，2 行
                        workgroupSize: [8, 4, 1],
                        tileInner: 32,
                    },
                ];
            } else {
                candidates = [
                    {
                        useVec4: false,
                        workPerThread: [2, 1, 1],
                        workgroupSize: [8, 8, 1],
                        tileInner: 16,
                    },
                    {
                        useVec4: false,
                        workPerThread: [4, 1, 1],
                        workgroupSize: [8, 8, 1],
                        tileInner: 16,
                    },
                ];
            }
            break;

        case 'tall':
            // 高矩阵特化 (N <= 8)
            // 每个线程处理多行，少量列
            if (canUseVec4) {
                candidates = [
                    {
                        useVec4: true,
                        workPerThread: [4, 4, 1],
                        workgroupSize: [8, 8, 1],
                        tileInner: 32,
                    },
                ];
            } else {
                candidates = [
                    {
                        useVec4: false,
                        workPerThread: [1, 4, 1],
                        workgroupSize: [8, 8, 1],
                        tileInner: 16,
                    },
                    {
                        useVec4: false,
                        workPerThread: [2, 4, 1],
                        workgroupSize: [8, 8, 1],
                        tileInner: 16,
                    },
                ];
            }
            break;

        case 'small':
            // 小矩阵，使用中等配置
            if (canUseVec4) {
                candidates = [
                    {
                        useVec4: true,
                        workPerThread: [4, 4, 1],
                        workgroupSize: [8, 8, 1],
                        tileInner: 32,
                    },
                    {
                        useVec4: true,
                        workPerThread: [4, 2, 1],
                        workgroupSize: [8, 8, 1],
                        tileInner: 32,
                    },
                ];
            } else {
                candidates = [
                    {
                        useVec4: false,
                        workPerThread: [2, 2, 1],
                        workgroupSize: [8, 8, 1],
                        tileInner: 32,
                    },
                    {
                        useVec4: false,
                        workPerThread: [1, 1, 1],
                        workgroupSize: [8, 8, 1],
                        tileInner: 32,
                    },
                ];
            }
            break;

        case 'medium':
            // 中等矩阵，标准配置
            if (canUseVec4) {
                candidates = [
                    {
                        useVec4: true,
                        workPerThread: [4, 4, 1],
                        workgroupSize: [8, 8, 1],
                        tileInner: 32,
                    },
                ];
            } else {
                candidates = [
                    {
                        useVec4: false,
                        workPerThread: [4, 4, 1],
                        workgroupSize: [8, 8, 1],
                        tileInner: 32,
                    },
                    {
                        useVec4: false,
                        workPerThread: [2, 2, 1],
                        workgroupSize: [8, 8, 1],
                        tileInner: 32,
                    },
                ];
            }
            break;

        case 'large':
            // 大矩阵，使用激进配置
            if (canUseVec4) {
                // FP16 可以使用更大的 tile（更少的 shared memory 压力）
                if (dtype === 'float16') {
                    candidates = [
                        {
                            useVec4: true,
                            workPerThread: [4, 8, 1],  // 更多行
                            workgroupSize: [8, 8, 1],
                            tileInner: 32,
                        },
                        {
                            useVec4: true,
                            workPerThread: [4, 4, 1],
                            workgroupSize: [8, 8, 1],
                            tileInner: 64,  // 更大的 K 分块
                        },
                    ];
                } else {
                    candidates = [
                        {
                            useVec4: true,
                            workPerThread: [4, 4, 1],
                            workgroupSize: [8, 8, 1],
                            tileInner: 32,
                        },
                    ];
                }
            } else {
                candidates = [
                    {
                        useVec4: false,
                        workPerThread: [4, 4, 1],
                        workgroupSize: [8, 8, 1],
                        tileInner: 32,
                    },
                    {
                        useVec4: false,
                        workPerThread: [2, 2, 1],
                        workgroupSize: [16, 16, 1],  // 更大的 workgroup
                        tileInner: 32,
                    },
                ];
            }
            break;
    }

    // 过滤出有效配置
    const validConfigs = candidates.filter(c =>
        isValidTileConfig(c, bytesPerElement, limits)
    );

    if (validConfigs.length > 0) {
        // 返回第一个有效配置（按优先级排序）
        return validConfigs[0];
    }

    // 终极回退：最保守的配置
    const fallbackConfig: TileConfig = {
        useVec4: false,
        workPerThread: [1, 1, 1],
        workgroupSize: [8, 8, 1],
        tileInner: 32,
    };

    // 如果回退配置也无效，进一步降级
    if (!isValidTileConfig(fallbackConfig, bytesPerElement, limits)) {
        return {
            useVec4: false,
            workPerThread: [1, 1, 1],
            workgroupSize: [4, 4, 1],
            tileInner: 16,
        };
    }

    return fallbackConfig;
}

/**
 * 计算 dispatch 维度
 * 
 * @param M 输出行数
 * @param N 输出列数
 * @param batchSize batch 大小
 * @param config Tile 配置
 */
export function calculateDispatchDimensions(
    M: number,
    N: number,
    batchSize: number,
    config: TileConfig
): [number, number, number] {
    const [colPerThread, rowPerThread, batchPerThread] = config.workPerThread;
    const [wgX, wgY, wgZ] = config.workgroupSize;

    // 每个工作组处理的输出尺寸
    const outputPerWgX = wgX * colPerThread; // 列方向
    const outputPerWgY = wgY * rowPerThread; // 行方向
    const outputPerWgZ = wgZ * batchPerThread; // batch 方向

    // 需要的工作组数量
    const dispatchX = Math.ceil(N / outputPerWgX);
    const dispatchY = Math.ceil(M / outputPerWgY);
    const dispatchZ = Math.ceil(batchSize / outputPerWgZ);

    return [dispatchX, dispatchY, dispatchZ];
}

/**
 * 获取 Shared Memory 大小
 * 
 * 用于验证配置不会超出设备限制
 * WebGPU 最大 workgroup storage 是 16KB (16384 bytes)
 * 
 * @param config Tile 配置
 * @param bytesPerElement 每个元素的字节数
 */
export function calculateSharedMemoryBytes(
    config: TileConfig,
    bytesPerElement: number
): number {
    const [wgX, wgY, _wgZ] = config.workgroupSize;
    const [colPerThread, rowPerThread, _batchPerThread] = config.workPerThread;
    const { tileInner, useVec4 } = config;

    // Tile A: [tileAOuter, tileInner] 或 vec4 压缩
    const tileAOuter = wgY * rowPerThread;

    // Tile B: [tileInner, tileBOuter] 或 vec4 压缩
    const tileBOuter = wgX * colPerThread;

    if (useVec4) {
        // Vec4 存储时，列方向压缩 4 倍
        const tileBElements = tileInner * (tileBOuter / 4);
        const tileAElements = tileAOuter * (tileInner / 4);
        return (tileAElements + tileBElements) * bytesPerElement * 4; // vec4 占 4 个元素
    } else {
        // 标量存储
        const tileAElements = tileAOuter * tileInner;
        const tileBElements = tileInner * tileBOuter;
        return (tileAElements + tileBElements) * bytesPerElement;
    }
}
