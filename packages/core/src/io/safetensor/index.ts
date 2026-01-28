/**
 * Safetensor 公开 API
 */

export type {
    SafetensorsDType,
    SafetensorLayer,
    SafetensorFile,
    SafetensorGroup,
    SafetensorsHeader,
    SafetensorsHeaderEntry,
    SafetensorsIndexJson,
} from './types';

export type {
    SAFETENSORS_DTYPE_MAP,
    SAFETENSORS_DTYPE_BYTES,
} from './dtypes';

export {
    mapSafetensorsDType,
    convertBF16toF32,
    createTypedArrayFromBuffer,
    getDtypeBytes,

} from './dtypes';

export {
    parseHeaderSize,
    parseJsonHeader,
    parseSafetensorFile,
    createSafetensorGroup,
    createSingleFileGroup,
} from './parser';

export { parseShardedIndex } from './sharded';

import { ByteSource, ResolvableByteSource } from '../source/types';
import { createByteSource, createResolvableByteSource } from '../source';
import { SafetensorGroup } from './types';
import { parseSafetensorFile, createSingleFileGroup } from './parser';
import { parseShardedIndex } from './sharded';

/**
 * 加载 Safetensor 文件或分片索引
 * 
 * 自动检测单文件 vs 分片模型:
 * - 如果路径以 `.index.json` 结尾，解析为分片模型
 * - 否则解析为单文件
 * 
 * @param source - URL、路径、ArrayBuffer 或 File
 * @param signal - 可选的取消信号
 * @returns SafetensorGroup（只包含 metadata 和 descriptor，不包含数据）
 * 
 * @example
 * // 单文件
 * const group = await loadSafetensor('./model.safetensors');
 * 
 * // 分片模型
 * const group = await loadSafetensor('./model.safetensors.index.json');
 * 
 * // Debug: 查看权重映射
 * group.dumpWeightMap();
 */
export async function loadSafetensor(
    source: string | ArrayBuffer | File,
    signal?: AbortSignal
): Promise<SafetensorGroup> {
    // 检测是否为分片模型
    const isSharded = typeof source === 'string' && source.endsWith('.index.json');

    if (isSharded) {
        // 分片模型需要 ResolvableByteSource
        const resolvableSource = createResolvableByteSource(source as string);
        return parseShardedIndex(resolvableSource, source as string, signal);
    } else {
        // 单文件
        const byteSource = createByteSource(source);
        const path = typeof source === 'string'
            ? source
            : (source instanceof File ? source.name : 'buffer');
        const file = await parseSafetensorFile(byteSource, path, signal);
        return createSingleFileGroup(file);
    }
}
