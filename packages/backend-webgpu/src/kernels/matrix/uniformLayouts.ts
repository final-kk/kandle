/**
 * Matmul Uniform Layouts
 * 
 * 结构化定义 Uniform Buffer 布局，避免手动管理索引
 * 
 * 每个 Layout 定义了：
 * - 字段名称和类型
 * - 自动计算的 offset 和 size
 * - 类型安全的写入方法
 */

// ============================================================
// 类型定义
// ============================================================

type UniformFieldType = 'u32' | 'f32';

interface UniformField {
    name: string;
    type: UniformFieldType;
}

interface UniformLayout {
    fields: UniformField[];
    size: number;
}

// ============================================================
// MM Uniform Layouts
// ============================================================

/**
 * MM Uniform Layout (无 InputC)
 * 
 * struct Uniforms {
 *     M: u32,          // 0
 *     K: u32,          // 4
 *     N: u32,          // 8
 *     offset_a: u32,   // 12
 *     offset_b: u32,   // 16
 *     offset_out: u32, // 20
 *     alpha: f32,      // 24
 *     beta: f32,       // 28
 * }
 */
export const MM_UNIFORM_LAYOUT: UniformLayout = {
    fields: [
        { name: 'M', type: 'u32' },
        { name: 'K', type: 'u32' },
        { name: 'N', type: 'u32' },
        { name: 'offset_a', type: 'u32' },
        { name: 'offset_b', type: 'u32' },
        { name: 'offset_out', type: 'u32' },
        { name: 'alpha', type: 'f32' },
        { name: 'beta', type: 'f32' },
    ],
    size: 32,
};

/**
 * MM GEMM Uniform Layout (有 InputC)
 * 
 * struct Uniforms {
 *     M: u32,           // 0
 *     K: u32,           // 4
 *     N: u32,           // 8
 *     offset_a: u32,    // 12
 *     offset_b: u32,    // 16
 *     offset_c: u32,    // 20
 *     offset_out: u32,  // 24
 *     c_shape_m: u32,   // 28
 *     c_shape_n: u32,   // 32
 *     alpha: f32,       // 36
 *     beta: f32,        // 40
 * }
 */
export const MM_GEMM_UNIFORM_LAYOUT: UniformLayout = {
    fields: [
        { name: 'M', type: 'u32' },
        { name: 'K', type: 'u32' },
        { name: 'N', type: 'u32' },
        { name: 'offset_a', type: 'u32' },
        { name: 'offset_b', type: 'u32' },
        { name: 'offset_c', type: 'u32' },
        { name: 'offset_out', type: 'u32' },
        { name: 'c_shape_m', type: 'u32' },
        { name: 'c_shape_n', type: 'u32' },
        { name: 'alpha', type: 'f32' },
        { name: 'beta', type: 'f32' },
    ],
    size: 44,
};

/**
 * BMM Uniform Layout (无 InputC) - 真正的 4D 支持
 * 
 * struct Uniforms {
 *     M: u32,               // 0
 *     K: u32,               // 4
 *     N: u32,               // 8
 *     batch_size: u32,      // 12
 *     offset_a: u32,        // 16
 *     offset_b: u32,        // 20
 *     offset_out: u32,      // 24
 *     ndim_a: u32,          // 28  - A 的维度数
 *     ndim_b: u32,          // 32  - B 的维度数
 *     _pad1: u32,           // 36
 *     _pad2: u32,           // 40
 *     _pad3: u32,           // 44
 *     // 完整 4D strides (每个 4 个 u32)
 *     strides_a: vec4<u32>, // 48-63: [s0, s1, s2, s3]
 *     strides_b: vec4<u32>, // 64-79: [s0, s1, s2, s3]
 *     alpha: f32,           // 80
 *     beta: f32,            // 84
 *     // --- 以下为兼容旧版本保留 ---
 *     stride_a_row: u32,    // 88  (deprecated)
 *     stride_a_col: u32,    // 92  (deprecated)
 *     stride_b_row: u32,    // 96  (deprecated)
 *     stride_b_col: u32,    // 100 (deprecated)
 *     batch_stride_a: u32,  // 104 (deprecated)
 *     batch_stride_b: u32,  // 108 (deprecated)
 * }
 */
export const BMM_UNIFORM_LAYOUT: UniformLayout = {
    fields: [
        { name: 'M', type: 'u32' },
        { name: 'K', type: 'u32' },
        { name: 'N', type: 'u32' },
        { name: 'batch_size', type: 'u32' },
        { name: 'offset_a', type: 'u32' },
        { name: 'offset_b', type: 'u32' },
        { name: 'offset_out', type: 'u32' },
        { name: 'ndim_a', type: 'u32' },
        { name: 'ndim_b', type: 'u32' },
        { name: '_pad1', type: 'u32' },
        { name: '_pad2', type: 'u32' },
        { name: '_pad3', type: 'u32' },
        // 4D strides for A (vec4 aligned)
        { name: 'strides_a_0', type: 'u32' },
        { name: 'strides_a_1', type: 'u32' },
        { name: 'strides_a_2', type: 'u32' },
        { name: 'strides_a_3', type: 'u32' },
        // 4D strides for B (vec4 aligned)
        { name: 'strides_b_0', type: 'u32' },
        { name: 'strides_b_1', type: 'u32' },
        { name: 'strides_b_2', type: 'u32' },
        { name: 'strides_b_3', type: 'u32' },
        { name: 'alpha', type: 'f32' },
        { name: 'beta', type: 'f32' },
        // 兼容旧版本
        { name: 'stride_a_row', type: 'u32' },
        { name: 'stride_a_col', type: 'u32' },
        { name: 'stride_b_row', type: 'u32' },
        { name: 'stride_b_col', type: 'u32' },
        { name: 'batch_stride_a', type: 'u32' },
        { name: 'batch_stride_b', type: 'u32' },
    ],
    size: 112,
};

/**
 * BMM GEMM Uniform Layout (有 InputC) - 真正的 4D 支持
 * 
 * struct Uniforms {
 *     M: u32,               // 0
 *     K: u32,               // 4
 *     N: u32,               // 8
 *     batch_size: u32,      // 12
 *     offset_a: u32,        // 16
 *     offset_b: u32,        // 20
 *     offset_c: u32,        // 24
 *     offset_out: u32,      // 28
 *     ndim_a: u32,          // 32
 *     ndim_b: u32,          // 36
 *     c_shape_m: u32,       // 40
 *     c_shape_n: u32,       // 44
 *     strides_a: vec4<u32>, // 48-63
 *     strides_b: vec4<u32>, // 64-79
 *     alpha: f32,           // 80
 *     beta: f32,            // 84
 *     // --- 兼容旧版本 ---
 *     stride_a_row: u32,    // 88 (deprecated)
 *     stride_a_col: u32,    // 92 (deprecated)
 *     stride_b_row: u32,    // 96 (deprecated)
 *     stride_b_col: u32,    // 100 (deprecated)
 *     batch_stride_a: u32,  // 104 (deprecated)
 *     batch_stride_b: u32,  // 108 (deprecated)
 * }
 */
export const BMM_GEMM_UNIFORM_LAYOUT: UniformLayout = {
    fields: [
        { name: 'M', type: 'u32' },
        { name: 'K', type: 'u32' },
        { name: 'N', type: 'u32' },
        { name: 'batch_size', type: 'u32' },
        { name: 'offset_a', type: 'u32' },
        { name: 'offset_b', type: 'u32' },
        { name: 'offset_c', type: 'u32' },
        { name: 'offset_out', type: 'u32' },
        { name: 'ndim_a', type: 'u32' },
        { name: 'ndim_b', type: 'u32' },
        { name: 'c_shape_m', type: 'u32' },
        { name: 'c_shape_n', type: 'u32' },
        // 4D strides for A
        { name: 'strides_a_0', type: 'u32' },
        { name: 'strides_a_1', type: 'u32' },
        { name: 'strides_a_2', type: 'u32' },
        { name: 'strides_a_3', type: 'u32' },
        // 4D strides for B
        { name: 'strides_b_0', type: 'u32' },
        { name: 'strides_b_1', type: 'u32' },
        { name: 'strides_b_2', type: 'u32' },
        { name: 'strides_b_3', type: 'u32' },
        { name: 'alpha', type: 'f32' },
        { name: 'beta', type: 'f32' },
        // 兼容旧版本
        { name: 'stride_a_row', type: 'u32' },
        { name: 'stride_a_col', type: 'u32' },
        { name: 'stride_b_row', type: 'u32' },
        { name: 'stride_b_col', type: 'u32' },
        { name: 'batch_stride_a', type: 'u32' },
        { name: 'batch_stride_b', type: 'u32' },
    ],
    size: 112,
};

// ============================================================
// Uniform Buffer Writer
// ============================================================

/**
 * 类型安全的 Uniform Buffer 写入器
 */
export class UniformBufferWriter {
    private buffer: ArrayBuffer;
    private u32View: Uint32Array;
    private f32View: Float32Array;
    private layout: UniformLayout;
    private fieldOffsets: Map<string, number>;

    constructor(layout: UniformLayout) {
        this.layout = layout;
        this.buffer = new ArrayBuffer(layout.size);
        this.u32View = new Uint32Array(this.buffer);
        this.f32View = new Float32Array(this.buffer);

        // 预计算字段 offset
        this.fieldOffsets = new Map();
        let offset = 0;
        for (const field of layout.fields) {
            this.fieldOffsets.set(field.name, offset);
            offset += 4; // 每个字段 4 bytes
        }
    }

    /**
     * 设置 u32 字段
     */
    setU32(name: string, value: number): this {
        const offset = this.fieldOffsets.get(name);
        if (offset === undefined) {
            throw new Error(`Unknown uniform field: ${name}`);
        }
        this.u32View[offset / 4] = value;
        return this;
    }

    /**
     * 设置 f32 字段
     */
    setF32(name: string, value: number): this {
        const offset = this.fieldOffsets.get(name);
        if (offset === undefined) {
            throw new Error(`Unknown uniform field: ${name}`);
        }
        this.f32View[offset / 4] = value;
        return this;
    }

    /**
     * 获取 ArrayBuffer
     */
    getBuffer(): ArrayBuffer {
        return this.buffer;
    }

    /**
     * 获取 buffer size
     */
    getSize(): number {
        return this.layout.size;
    }
}

// ============================================================
// 便捷函数
// ============================================================

/**
 * MM Uniform 数据接口
 */
export interface MmUniformData {
    M: number;
    K: number;
    N: number;
    offsetA: number;
    offsetB: number;
    offsetOut: number;
    alpha: number;
    beta: number;
}

/**
 * MM GEMM Uniform 数据接口
 */
export interface MmGemmUniformData extends MmUniformData {
    offsetC: number;
    cShapeM: number;
    cShapeN: number;
}

/**
 * BMM Uniform 数据接口 - 真正的 4D 支持
 */
export interface BmmUniformData extends MmUniformData {
    batchSize: number;
    ndimA: number;
    ndimB: number;
    /** 完整 4D strides: [s0, s1, s2, s3] (已 padding) */
    fullStridesA: readonly [number, number, number, number];
    fullStridesB: readonly [number, number, number, number];
    // 兼容旧版本
    strideARow: number;
    strideACol: number;
    strideBRow: number;
    strideBCol: number;
    batchStrideA: number;
    batchStrideB: number;
}

/**
 * BMM GEMM Uniform 数据接口
 */
export interface BmmGemmUniformData extends BmmUniformData {
    offsetC: number;
    cShapeM: number;
    cShapeN: number;
}

/**
 * 创建 MM Uniform Buffer
 */
export function createMmUniformBuffer(data: MmUniformData): ArrayBuffer {
    const writer = new UniformBufferWriter(MM_UNIFORM_LAYOUT);
    return writer
        .setU32('M', data.M)
        .setU32('K', data.K)
        .setU32('N', data.N)
        .setU32('offset_a', data.offsetA)
        .setU32('offset_b', data.offsetB)
        .setU32('offset_out', data.offsetOut)
        .setF32('alpha', data.alpha)
        .setF32('beta', data.beta)
        .getBuffer();
}

/**
 * 创建 MM GEMM Uniform Buffer
 */
export function createMmGemmUniformBuffer(data: MmGemmUniformData): ArrayBuffer {
    const writer = new UniformBufferWriter(MM_GEMM_UNIFORM_LAYOUT);
    return writer
        .setU32('M', data.M)
        .setU32('K', data.K)
        .setU32('N', data.N)
        .setU32('offset_a', data.offsetA)
        .setU32('offset_b', data.offsetB)
        .setU32('offset_c', data.offsetC)
        .setU32('offset_out', data.offsetOut)
        .setU32('c_shape_m', data.cShapeM)
        .setU32('c_shape_n', data.cShapeN)
        .setF32('alpha', data.alpha)
        .setF32('beta', data.beta)
        .getBuffer();
}

/**
 * 创建 BMM Uniform Buffer - 真正的 4D 支持
 */
export function createBmmUniformBuffer(data: BmmUniformData): ArrayBuffer {
    const writer = new UniformBufferWriter(BMM_UNIFORM_LAYOUT);
    return writer
        .setU32('M', data.M)
        .setU32('K', data.K)
        .setU32('N', data.N)
        .setU32('batch_size', data.batchSize)
        .setU32('offset_a', data.offsetA)
        .setU32('offset_b', data.offsetB)
        .setU32('offset_out', data.offsetOut)
        .setU32('ndim_a', data.ndimA)
        .setU32('ndim_b', data.ndimB)
        .setU32('_pad1', 0)
        .setU32('_pad2', 0)
        .setU32('_pad3', 0)
        // 4D strides for A
        .setU32('strides_a_0', data.fullStridesA[0])
        .setU32('strides_a_1', data.fullStridesA[1])
        .setU32('strides_a_2', data.fullStridesA[2])
        .setU32('strides_a_3', data.fullStridesA[3])
        // 4D strides for B
        .setU32('strides_b_0', data.fullStridesB[0])
        .setU32('strides_b_1', data.fullStridesB[1])
        .setU32('strides_b_2', data.fullStridesB[2])
        .setU32('strides_b_3', data.fullStridesB[3])
        .setF32('alpha', data.alpha)
        .setF32('beta', data.beta)
        // 兼容旧版本
        .setU32('stride_a_row', data.strideARow)
        .setU32('stride_a_col', data.strideACol)
        .setU32('stride_b_row', data.strideBRow)
        .setU32('stride_b_col', data.strideBCol)
        .setU32('batch_stride_a', data.batchStrideA)
        .setU32('batch_stride_b', data.batchStrideB)
        .getBuffer();
}

/**
 * 创建 BMM GEMM Uniform Buffer - 真正的 4D 支持
 */
export function createBmmGemmUniformBuffer(data: BmmGemmUniformData): ArrayBuffer {
    const writer = new UniformBufferWriter(BMM_GEMM_UNIFORM_LAYOUT);
    return writer
        .setU32('M', data.M)
        .setU32('K', data.K)
        .setU32('N', data.N)
        .setU32('batch_size', data.batchSize)
        .setU32('offset_a', data.offsetA)
        .setU32('offset_b', data.offsetB)
        .setU32('offset_c', data.offsetC)
        .setU32('offset_out', data.offsetOut)
        .setU32('ndim_a', data.ndimA)
        .setU32('ndim_b', data.ndimB)
        .setU32('c_shape_m', data.cShapeM)
        .setU32('c_shape_n', data.cShapeN)
        // 4D strides for A
        .setU32('strides_a_0', data.fullStridesA[0])
        .setU32('strides_a_1', data.fullStridesA[1])
        .setU32('strides_a_2', data.fullStridesA[2])
        .setU32('strides_a_3', data.fullStridesA[3])
        // 4D strides for B
        .setU32('strides_b_0', data.fullStridesB[0])
        .setU32('strides_b_1', data.fullStridesB[1])
        .setU32('strides_b_2', data.fullStridesB[2])
        .setU32('strides_b_3', data.fullStridesB[3])
        .setF32('alpha', data.alpha)
        .setF32('beta', data.beta)
        // 兼容旧版本
        .setU32('stride_a_row', data.strideARow)
        .setU32('stride_a_col', data.strideACol)
        .setU32('stride_b_row', data.strideBRow)
        .setU32('stride_b_col', data.strideBCol)
        .setU32('batch_stride_a', data.batchStrideA)
        .setU32('batch_stride_b', data.batchStrideB)
        .getBuffer();
}

