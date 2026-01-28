/**
 * WebByteSource - 基于 fetch API 的 ByteSource 实现
 * 
 * 特性:
 * - 使用 Range request 进行分片读取
 * - 使用 URL API 正确处理相对路径 (支持 ../ 和 ./)
 * - 支持 AbortSignal 取消请求
 * - 自动 fallback 到全量下载 (当服务器不支持 Range 时)
 */

import { ByteSource, ResolvableByteSource } from './types';

export class WebByteSource implements ResolvableByteSource {
    private _size: number | null = null;
    private readonly _url: URL;

    constructor(url: string | URL) {
        // 使用 URL API 正确处理相对路径和特殊字符
        this._url = typeof url === 'string'
            ? new URL(url, typeof location !== 'undefined' ? location.href : undefined)
            : url;
    }

    /**
     * 获取底层 URL (用于调试)
     */
    get url(): string {
        return this._url.href;
    }

    async read(offset: number, length: number, signal?: AbortSignal): Promise<ArrayBuffer> {
        const response = await fetch(this._url.href, {
            headers: {
                'Range': `bytes=${offset}-${offset + length - 1}`,
            },
            signal,
        });

        if (response.status === 206) {
            // Partial Content - Range 请求成功
            return response.arrayBuffer();
        } else if (response.status === 200) {
            // 服务器不支持 Range，需要读取全部后切片
            const buffer = await response.arrayBuffer();
            return buffer.slice(offset, offset + length);
        } else {
            throw new Error(`HTTP error: ${response.status} ${response.statusText}`);
        }
    }

    async size(signal?: AbortSignal): Promise<number> {
        if (this._size !== null) return this._size;

        // HEAD 请求获取 Content-Length
        const response = await fetch(this._url.href, { method: 'HEAD', signal });
        const contentLength = response.headers.get('Content-Length');
        if (contentLength) {
            this._size = parseInt(contentLength, 10);
            return this._size;
        }

        throw new Error('Cannot determine file size: Content-Length header missing');
    }

    resolve(relativePath: string): WebByteSource {
        // 使用 URL API 正确处理相对路径 (包括 ../ 和 ./)
        return new WebByteSource(new URL(relativePath, this._url));
    }

    close(): void {
        // Web 无需显式释放资源
        // fetch 请求由浏览器管理
    }
}
