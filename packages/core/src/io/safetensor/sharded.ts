/**
 * Safetensors 分片模型解析器
 * 
 * 解析 model.safetensors.index.json 并加载所有分片文件
 */

import { ResolvableByteSource } from '../source/types';
import { SafetensorGroup, SafetensorFile, SafetensorLayer, SafetensorsIndexJson } from './types';
import { parseSafetensorFile, createSafetensorGroup } from './parser';

/**
 * 解析 index.json (分片模型)
 * 
 * @param source - 可解析相对路径的数据源
 * @param path - index.json 路径 (用于标识)
 * @param signal - 可选的取消信号
 * @returns SafetensorGroup 对象
 */
export async function parseShardedIndex(
    source: ResolvableByteSource,
    path: string,
    signal?: AbortSignal
): Promise<SafetensorGroup> {
    // 1. 读取完整 index.json
    const size = await source.size(signal);
    const buffer = await source.read(0, size, signal);
    const decoder = new TextDecoder('utf-8');

    let index: SafetensorsIndexJson;
    try {
        index = JSON.parse(decoder.decode(buffer));
    } catch (e) {
        throw new Error(`Invalid index.json: ${(e as Error).message}`);
    }

    // 2. 验证 index.json 结构
    if (!index.weight_map || typeof index.weight_map !== 'object') {
        throw new Error('Invalid index.json: weight_map must be an object');
    }

    // 3. 收集所有分片文件名
    const shardNames = new Set<string>(Object.values(index.weight_map));

    // 4. 并行解析每个分片文件
    const files = new Map<string, SafetensorFile>();
    const loadPromises: Promise<void>[] = [];

    for (const shardName of shardNames) {
        const promise = (async () => {
            // 检查取消信号
            if (signal?.aborted) {
                throw new Error("Aborted");
            }

            const shardSource = source.resolve(shardName);
            const shardFile = await parseSafetensorFile(shardSource, shardName, signal);
            files.set(shardName, shardFile);
        })();
        loadPromises.push(promise);
    }

    await Promise.all(loadPromises);

    // 5. 构建扁平 layer 视图 (按 weight_map 的顺序)
    const layers = new Map<string, SafetensorLayer>();
    for (const [layerName, shardName] of Object.entries(index.weight_map)) {
        const file = files.get(shardName);
        if (!file) {
            throw new Error(`Shard file not found: ${shardName}`);
        }

        const layer = file.layers.get(layerName);
        if (!layer) {
            throw new Error(`Layer not found in shard ${shardName}: ${layerName}`);
        }

        layers.set(layerName, layer);
    }

    // 6. 提取元数据
    const metadata: Record<string, string> = {};
    if (index.metadata) {
        for (const [key, value] of Object.entries(index.metadata)) {
            if (typeof value === 'string') {
                metadata[key] = value;
            }
        }
    }

    return createSafetensorGroup({
        sharded: true,
        metadata,
        totalSize: index.metadata?.total_size,
        files,
        layers,
    });
}
