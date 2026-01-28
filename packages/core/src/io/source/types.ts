/**
 * ByteSource - 平台无关的字节读取抽象
 * 
 * 设计要点:
 * - 支持 Range 读取（分片加载）
 * - 延迟获取 size（避免不必要的 HEAD 请求）
 * - 可释放资源
 * - 支持 AbortSignal 取消请求
 */
export interface ByteSource {
    /** 
     * 读取指定范围的字节
     * 
     * @param offset - 起始偏移（字节）
     * @param length - 读取长度（字节）
     * @param signal - 可选的 AbortSignal，用于取消请求
     * @returns 原始字节数据
     */
    read(offset: number, length: number, signal?: AbortSignal): Promise<ArrayBuffer>;

    /** 
     * 获取总大小
     * 
     * Web: 可能需要 HEAD 请求
     * Node: 使用 fs.stat
     * @param signal - 可选的 AbortSignal，用于取消请求
     */
    size(signal?: AbortSignal): Promise<number>;

    /** 
     * 释放资源
     * 
     * Web: 释放缓存
     * Node: 关闭文件句柄
     */
    close(): void;
}

/**
 * 可解析相对路径的 ByteSource
 * 
 * 用于分片场景：从 index.json 解析出各分片文件路径
 */
export interface ResolvableByteSource extends ByteSource {
    /** 
     * 解析相对路径，返回新的 ByteSource
     * 
     * @param relativePath - 相对于当前源的路径
     * @example
     * // 当前源: https://example.com/model/model.safetensors.index.json
     * source.resolve('model-00001-of-00002.safetensors')
     * // → https://example.com/model/model-00001-of-00002.safetensors
     */
    resolve(relativePath: string): ByteSource;
}
