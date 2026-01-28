/**
 * Safetensors 单文件解析器
 * 
 * 只读取 header，不读取数据
 */

import { ByteSource } from '../source/types';
import {
    SafetensorFile,
    SafetensorLayer,
    SafetensorGroup,
    SafetensorsHeader,
    SafetensorsHeaderEntry,
    SafetensorsDType,
} from './types';
import { mapSafetensorsDType, SAFETENSORS_DTYPE_BYTES } from './dtypes';

// ============================================================================
// Header Parsing
// ============================================================================

/**
 * 解析 header size (前 8 字节，little-endian u64)
 * 
 * @param buffer - 前 8 字节
 * @returns header 大小
 */
export function parseHeaderSize(buffer: ArrayBuffer): number {
    const view = new DataView(buffer);
    // 使用 BigInt 读取完整的 u64
    const low = view.getUint32(0, true);
    const high = view.getUint32(4, true);

    // 对于合理的 header 大小，high 应该为 0
    if (high !== 0) {
        throw new Error(`Header size too large: exceeds safe integer limit (high bytes: ${high})`);
    }

    // 安全限制: header 不应超过 100MB
    const MAX_HEADER_SIZE = 100 * 1024 * 1024;
    if (low > MAX_HEADER_SIZE) {
        throw new Error(`Header size exceeds safe limit: ${low} bytes (max: ${MAX_HEADER_SIZE})`);
    }

    return low;
}

/**
 * 解析 JSON header
 * 
 * @param buffer - header 字节数据
 * @returns 解析后的 header 对象
 */
export function parseJsonHeader(buffer: ArrayBuffer): SafetensorsHeader {
    const decoder = new TextDecoder('utf-8');
    const jsonStr = decoder.decode(buffer);

    try {
        return JSON.parse(jsonStr) as SafetensorsHeader;
    } catch (e) {
        throw new Error(`Invalid JSON in safetensors header: ${(e as Error).message}`);
    }
}

/**
 * 检查是否为 tensor entry (而非 __metadata__)
 */
export function isTensorEntry(entry: unknown): entry is SafetensorsHeaderEntry {
    if (typeof entry !== 'object' || entry === null) return false;
    const obj = entry as Record<string, unknown>;
    return (
        typeof obj.dtype === 'string' &&
        Array.isArray(obj.shape) &&
        Array.isArray(obj.data_offsets) &&
        obj.data_offsets.length === 2
    );
}

// ============================================================================
// File Parsing
// ============================================================================

/**
 * 解析单个 .safetensors 文件
 * 只读取 header，不读取数据
 * 
 * @param source - 数据源
 * @param path - 文件路径 (用于标识)
 * @param signal - 可选的取消信号
 * @returns SafetensorFile 对象
 */
export async function parseSafetensorFile(
    source: ByteSource,
    path: string,
    signal?: AbortSignal
): Promise<SafetensorFile> {
    // 1. 读取 header size (8 bytes)
    const sizeBuffer = await source.read(0, 8, signal);
    if (sizeBuffer.byteLength < 8) {
        throw new Error('Invalid safetensors: file too small to contain header size');
    }
    const headerSize = parseHeaderSize(sizeBuffer);

    // 2. 验证 header size
    if (headerSize === 0) {
        throw new Error('Invalid safetensors: header size is 0');
    }

    // 3. 读取 JSON header
    const headerBuffer = await source.read(8, headerSize, signal);
    if (headerBuffer.byteLength < headerSize) {
        throw new Error('Invalid safetensors: header data incomplete');
    }
    const header = parseJsonHeader(headerBuffer);

    // 4. 提取 metadata
    const metadata = (header.__metadata__ ?? {}) as Record<string, string>;

    // 5. 计算数据区偏移
    const dataOffset = 8 + headerSize;

    // 6. 创建 file 对象骨架 (layers 稍后填充)
    const layers = new Map<string, SafetensorLayer>();

    const file: SafetensorFile = {
        path,
        metadata,
        layers,
        source,
        dataOffset,
    };

    // 7. 解析每个 tensor entry
    for (const [name, entry] of Object.entries(header)) {
        if (name === '__metadata__') continue;
        if (!isTensorEntry(entry)) {
            // 跳过非 tensor entry
            continue;
        }

        // 验证 dtype
        const stDtype = entry.dtype as SafetensorsDType;
        if (!(stDtype in SAFETENSORS_DTYPE_BYTES)) {
            throw new Error(`Unsupported safetensors dtype: ${stDtype}`);
        }

        // 计算元素数量
        const numel = entry.shape.reduce((a, b) => a * b, 1);
        const byteSize = entry.data_offsets[1] - entry.data_offsets[0];

        // 验证数据大小
        const expectedBytes = numel * SAFETENSORS_DTYPE_BYTES[stDtype];
        if (numel > 0 && byteSize !== expectedBytes) {
            throw new Error(
                `Data size mismatch for ${name}: expected ${expectedBytes} bytes, got ${byteSize} bytes`
            );
        }

        const layer: SafetensorLayer = {
            name,
            dtype: mapSafetensorsDType(stDtype),
            shape: entry.shape,
            originalDtype: stDtype,
            file,
            dataOffsets: entry.data_offsets,
            byteSize,
            numel,
        };

        layers.set(name, layer);
    }

    return file;
}

// ============================================================================
// SafetensorGroup Factory
// ============================================================================

interface SafetensorGroupOptions {
    sharded: boolean;
    metadata: Record<string, string>;
    totalSize?: number;
    files: Map<string, SafetensorFile>;
    layers: Map<string, SafetensorLayer>;
}

/**
 * 创建 SafetensorGroup 对象
 */
export function createSafetensorGroup(options: SafetensorGroupOptions): SafetensorGroup {
    const { sharded, metadata, totalSize, files, layers } = options;

    return {
        sharded,
        metadata,
        totalSize,
        files,
        layers,

        getLayer(name: string): SafetensorLayer | undefined {
            return layers.get(name);
        },

        hasLayer(name: string): boolean {
            return layers.has(name);
        },

        dumpWeightMap(): void {
            console.log('=== Weight Map ===');
            console.log(`Total layers: ${layers.size}`);
            console.log(`Sharded: ${sharded}`);
            if (sharded) {
                console.log(`Files: ${files.size}`);
            }
            console.log('');

            // 找到最长的 name 用于对齐
            let maxNameLen = 0;
            for (const name of layers.keys()) {
                maxNameLen = Math.max(maxNameLen, name.length);
            }

            for (const [name, layer] of layers) {
                const paddedName = name.padEnd(maxNameLen + 2);
                const shapeStr = `[${layer.shape.join(', ')}]`.padEnd(20);
                const dtypeStr = layer.dtype.padEnd(10);
                const filename = layer.file.path;
                console.log(`${paddedName}${shapeStr}${dtypeStr}${filename}`);
            }
        },

        close(): void {
            for (const file of files.values()) {
                file.source.close();
            }
        },
    };
}

/**
 * 从单个文件创建 SafetensorGroup
 */
export function createSingleFileGroup(file: SafetensorFile): SafetensorGroup {
    const files = new Map<string, SafetensorFile>();
    files.set(file.path, file);

    return createSafetensorGroup({
        sharded: false,
        metadata: file.metadata,
        files,
        layers: new Map(file.layers),
    });
}
