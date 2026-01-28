export interface IAllocator {
    alloc(size: number): ArrayBuffer;
    free(buffer: ArrayBuffer): void;
}
