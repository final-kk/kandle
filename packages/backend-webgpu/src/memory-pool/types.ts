/**
 * GPU Memory Pool Types
 * 
 * 核心类型定义，用于 Arena-based GPU 内存池
 */

// ============================================================================
// Buffer Usage Presets
// ============================================================================

/**
 * 预定义的常用 Buffer Usage 组合
 * 
 * WebGPU 中 Buffer 的 usage 标志决定了它能参与哪些操作。
 * 不同 usage 的 Buffer 不能混用，因此需要按 usage 分池。
 */
export const BufferUsagePreset = {
    /** 计算用途（最常见）：Shader 读写 + 拷贝 */
    STORAGE: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,

    /** Uniform 参数：只读 + 拷贝写入 */
    UNIFORM: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,

    /** Indirect Dispatch：间接调度命令 */
    INDIRECT: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,

    /** Staging (读回)：用于 mapAsync 读取 GPU 数据 */
    STAGING_READ: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,

    /** Staging (写入)：用于 mapAsync 写入数据到 GPU */
    STAGING_WRITE: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
} as const;

export type BufferUsagePresetType = typeof BufferUsagePreset[keyof typeof BufferUsagePreset];

// ============================================================================
// Allocation Record (SoA Layout)
// ============================================================================

/**
 * AllocationRecord 的逻辑结构
 * 
 * 实际存储使用 Struct of Arrays (SoA) 以避免 JS 对象开销：
 * - 每个字段使用独立的 TypedArray
 * - 通过 allocId 索引访问
 */
export interface AllocationRecord {
    /** 所属 Arena 的 ID */
    arenaId: number;
    /** 在 Arena 内的字节偏移 */
    offset: number;
    /** 分配的字节大小 */
    size: number;
    /** 引用计数 */
    refCount: number;
    /** 最后使用的 GPU fence epoch */
    fenceEpoch: number;
    /** 代际计数（用于检测 use-after-free） */
    generation: number;
}

// ============================================================================
// Arena Configuration
// ============================================================================

/**
 * Arena 大小配置
 */
export interface ArenaConfig {
    /** 小型 Arena (64MB 默认) */
    small: number;
    /** 中型 Arena (256MB 默认) */
    medium: number;
    /** 大型 Arena (1GB 默认) */
    large: number;
}

/**
 * 默认 Arena 配置
 */
export const DEFAULT_ARENA_CONFIG: ArenaConfig = {
    small: 64 * 1024 * 1024,      // 64 MB
    medium: 256 * 1024 * 1024,    // 256 MB
    large: 1024 * 1024 * 1024,    // 1 GB
};

// ============================================================================
// Size Class Configuration
// ============================================================================

/**
 * Size Class 配置
 * 
 * 使用 36 级 Size Classes，覆盖 256B - 2GB 范围
 * 每级 = 2^(8 + level) bytes
 * 
 * Level 0:  256 B
 * Level 1:  512 B
 * Level 2:  1 KB
 * ...
 * Level 35: 2 GB
 */
export const SIZE_CLASS_CONFIG = {
    /** Size Class 数量 */
    NUM_CLASSES: 36,
    /** 最小分配大小 (256 bytes) */
    MIN_SIZE: 256,
    /** 最小 size class 的 log2 值 */
    MIN_SIZE_LOG2: 8,
} as const;

// ============================================================================
// Debug Mode
// ============================================================================

/**
 * 是否启用 Debug 模式
 * 
 * Debug 模式下会启用：
 * - AllocId Token 校验
 * - 泄漏检测
 * - 详细日志
 */
export const DEBUG_MODE: boolean = (() => {
    // 浏览器环境：默认开启 debug
    // 可通过 globalThis.__NN_KIT_DEBUG__ = false 关闭
    if (typeof globalThis !== 'undefined' && '__NN_KIT_DEBUG__' in globalThis) {
        return !!(globalThis as Record<string, unknown>).__NN_KIT_DEBUG__;
    }
    // 默认关闭
    return false;
})();

// ============================================================================
// Errors
// ============================================================================

/**
 * GPU OOM 错误
 */
export class OutOfMemoryError extends Error {
    constructor(message: string) {
        super(message);
        this.name = 'OutOfMemoryError';
    }
}

/**
 * 无效分配 ID 错误（use-after-free）
 */
export class InvalidAllocationError extends Error {
    constructor(allocId: number, reason: string) {
        super(`Invalid allocation ID ${allocId}: ${reason}`);
        this.name = 'InvalidAllocationError';
    }
}
