/**
 * Safetensors 类型定义
 * 
 * 核心设计: 只持有 metadata 和 descriptor，不持有数据
 */

import { DType } from '@kandle/types';
import { ByteSource } from '../source/types';

// ============================================================================
// Safetensors DType
// ============================================================================

/**
 * Safetensors 原始 dtype 字符串
 */
export type SafetensorsDType =
    | 'F64' | 'F32' | 'F16' | 'BF16'
    | 'I64' | 'I32' | 'I16' | 'I8'
    | 'U64' | 'U32' | 'U16' | 'U8'
    | 'BOOL';

// ============================================================================
// SafetensorLayer
// ============================================================================

/**
 * 单个 Layer 的描述符
 * 
 * 注意: 不持有数据，只持有定位信息
 */
export interface SafetensorLayer {
    /** 原始键名 (e.g., "model.layers.0.self_attn.q_proj.weight") */
    readonly name: string;

    /** 转换后的 NN-Kit dtype */
    readonly dtype: DType;

    /** 形状 */
    readonly shape: readonly number[];

    /** 原始 safetensor dtype (用于 BF16 等转换) */
    readonly originalDtype: SafetensorsDType;

    /** 所属文件引用 */
    readonly file: SafetensorFile;

    /** 
     * 在文件数据区内的字节偏移 [begin, end)
     * 相对于 file.dataOffset
     */
    readonly dataOffsets: readonly [number, number];

    /** 数据大小（字节） */
    readonly byteSize: number;

    /** 元素数量 */
    readonly numel: number;
}

// ============================================================================
// SafetensorFile
// ============================================================================

/**
 * 单个 .safetensors 文件
 */
export interface SafetensorFile {
    /** 文件标识（路径或 URL） */
    readonly path: string;

    /** 文件级元数据 (__metadata__ 字段) */
    readonly metadata: Readonly<Record<string, string>>;

    /** 本文件包含的所有 layer */
    readonly layers: ReadonlyMap<string, SafetensorLayer>;

    /** 底层数据源 */
    readonly source: ByteSource;

    /** 数据区起始偏移 (header_size + 8) */
    readonly dataOffset: number;
}

// ============================================================================
// SafetensorGroup
// ============================================================================

/**
 * Safetensor 组
 * 
 * 可能是单个文件，也可能是 index.json + 多个分片
 */
export interface SafetensorGroup {
    /** 是否分片模型 */
    readonly sharded: boolean;

    /** 组级元数据 (来自 index.json 或单文件) */
    readonly metadata: Readonly<Record<string, string>>;

    /** 总大小（字节，来自 index.json metadata.total_size） */
    readonly totalSize?: number;

    /** 所有文件 (key = 文件名) */
    readonly files: ReadonlyMap<string, SafetensorFile>;

    /** 
     * 扁平化的 layer 视图（跨所有文件）
     * 用于统一访问，无需关心分片
     */
    readonly layers: ReadonlyMap<string, SafetensorLayer>;

    /**
     * 获取指定 layer
     * 
     * @param name - layer 名称
     * @returns Layer 描述符，不存在则返回 undefined
     */
    getLayer(name: string): SafetensorLayer | undefined;

    /**
     * 检查是否包含指定 layer
     */
    hasLayer(name: string): boolean;

    /**
     * Debug: 打印权重映射表
     * 
     * 输出格式:
     * model.embed_tokens.weight    [151936, 896]  float16  model-00001-of-00002.safetensors
     */
    dumpWeightMap(): void;

    /**
     * 释放所有资源
     */
    close(): void;
}

// ============================================================================
// Internal Types (解析用)
// ============================================================================

/**
 * Safetensors JSON header 中的 tensor entry
 */
export interface SafetensorsHeaderEntry {
    dtype: SafetensorsDType;
    shape: number[];
    data_offsets: [number, number];
}

/**
 * Safetensors JSON header
 */
export interface SafetensorsHeader {
    __metadata__?: Record<string, string>;
    [key: string]: SafetensorsHeaderEntry | Record<string, string> | undefined;
}

/**
 * index.json 结构
 */
export interface SafetensorsIndexJson {
    metadata?: {
        total_size?: number;
        [key: string]: unknown;
    };
    weight_map: Record<string, string>;
}
