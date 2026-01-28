import { DeviceNameEnum } from "./base";

export type StorageBufferType = ArrayBufferLike | GPUBuffer;

export interface IStorage {
    readonly storageId: number;
    /** 
     * 底层 buffer 引用
     * 
     * 在 Memory Pool 架构下，这可能是一个共享的 Arena buffer，
     * 需要配合 bufferOffset 使用来定位实际数据位置。
     */
    readonly buffer: StorageBufferType;
    /**
     * 数据在 buffer 中的字节偏移量
     * 
     * 对于独占 buffer 的 storage，这总是 0。
     * 对于 Arena-based Memory Pool，这是分配在 Arena 内的偏移位置。
     */
    readonly bufferOffset: number;
    /** 分配的字节大小 */
    readonly byteLength: number;
    readonly device: DeviceNameEnum;
    toRawDataAsync?(): Promise<ArrayBuffer>;
}
