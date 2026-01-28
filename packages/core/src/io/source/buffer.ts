/**
 * ArrayBufferByteSource - 基于 ArrayBuffer 的 ByteSource 实现
 * 
 * 用于从内存中已有的 ArrayBuffer 读取数据，
 * 主要用于测试或处理已下载的完整文件
 */

import { ByteSource } from './types';

export class ArrayBufferByteSource implements ByteSource {
    constructor(private readonly buffer: ArrayBuffer) { }

    async read(offset: number, length: number, _signal?: AbortSignal): Promise<ArrayBuffer> {
        // 使用 slice 创建副本，保证 TypedArray 内存对齐
        return this.buffer.slice(offset, offset + length);
    }

    async size(_signal?: AbortSignal): Promise<number> {
        return this.buffer.byteLength;
    }

    close(): void {
        // ArrayBuffer 由 GC 回收，无需显式释放
    }
}
