/**
 * ByteSource 模块入口
 * 
 * 提供统一的工厂函数和类型导出
 */

export type { ByteSource, ResolvableByteSource } from './types';
export { WebByteSource } from './web';
export { ArrayBufferByteSource } from './buffer';
export { FileByteSource } from './file';

import type { ByteSource, ResolvableByteSource } from './types';
import { WebByteSource } from './web';
import { ArrayBufferByteSource } from './buffer';
import { FileByteSource } from './file';

/**
 * 自动检测环境并创建合适的 ByteSource
 * 
 * @param source - URL 字符串、ArrayBuffer 或 File 对象
 * @returns ByteSource 实例
 * 
 * @example
 * // 从 URL 创建
 * const source = createByteSource('https://example.com/model.safetensors');
 * 
 * // 从 ArrayBuffer 创建
 * const source = createByteSource(buffer);
 * 
 * // 从 File 创建 (浏览器)
 * const source = createByteSource(file);
 */
export function createByteSource(source: string | ArrayBuffer | File): ByteSource {
    if (typeof source === 'string') {
        // URL 字符串 → WebByteSource
        return new WebByteSource(source);
    } else if (source instanceof ArrayBuffer) {
        return new ArrayBufferByteSource(source);
    } else if (typeof File !== 'undefined' && source instanceof File) {
        return new FileByteSource(source);
    }
    throw new Error(`Unsupported source type: ${typeof source}`);
}

/**
 * 创建可解析相对路径的 ByteSource
 * 
 * 仅适用于 URL 字符串，用于分片模型加载场景
 * 
 * @param source - URL 字符串
 * @returns ResolvableByteSource 实例
 */
export function createResolvableByteSource(source: string): ResolvableByteSource {
    return new WebByteSource(source);
}
