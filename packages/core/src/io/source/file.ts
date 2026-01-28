/**
 * FileByteSource - 基于浏览器 File API 的 ByteSource 实现
 * 
 * 用于处理用户通过 <input type="file"> 选择的文件
 */

import { ByteSource } from './types';

export class FileByteSource implements ByteSource {
    constructor(private readonly file: File) { }

    async read(offset: number, length: number, _signal?: AbortSignal): Promise<ArrayBuffer> {
        // 使用 Blob.slice 进行 Range 读取
        const blob = this.file.slice(offset, offset + length);
        return blob.arrayBuffer();
    }

    async size(_signal?: AbortSignal): Promise<number> {
        return this.file.size;
    }

    close(): void {
        // File 对象由 GC 回收，无需显式释放
    }
}
