import { DataTypeMap, DeviceNameEnum, DType, ITensorHandle, MemoryFormat, Shape, TensorData, TensorHandleSymbol } from "@kandle/types";
import { computeStrides, computeStridesForFormat, getTypedArrayCtor, GlobalIdManager, inferMemoryFormat, isContiguousStrides } from "@kandle/utils";
import { WebGPUStorage } from "./storage";
import { getGlobalDTypeResolver } from "./DTypeResolver";


export class WebGPUTensor<T extends DType> implements ITensorHandle {
    readonly dtype: T;
    readonly shape: Shape;
    readonly numel: number;
    readonly strides: number[];
    readonly offset: number;
    readonly id: number;
    readonly storage: WebGPUStorage;
    readonly memoryFormat: MemoryFormat;
    private _disposed: boolean = false;
    readonly [TensorHandleSymbol] = true;

    get buffer(): GPUBuffer {
        return this.storage.buffer;
    }

    /**
     * 私有构造函数，外部应使用静态工厂方法
     */
    private constructor(
        shape: Shape,
        dtype: T,
        storage: WebGPUStorage,
        strides: number[],
        offset: number,
        memoryFormat: MemoryFormat
    ) {
        this.dtype = dtype;
        this.shape = shape;
        this.numel = shape.reduce((acc, val) => acc * val, 1);
        this.strides = strides;
        this.offset = offset;
        this.storage = storage;
        this.memoryFormat = memoryFormat;
        this.id = GlobalIdManager.getNextTensorId();
    }

    /**
     * 创建新的 Tensor，分配新的 Storage
     */
    /**
     * 创建新的 Tensor，分配新的 Storage
     * 
     * @param shape - 张量形状
     * @param dtype - 数据类型
     * @param data - 可选的初始数据
     * @param memoryFormat - 内存格式 (默认 Contiguous)
     */
    static createNew<T extends DType>(
        shape: Shape,
        dtype: T,
        data?: DataTypeMap[T],
        memoryFormat: MemoryFormat = MemoryFormat.Contiguous
    ): WebGPUTensor<T> {
        const resolver = getGlobalDTypeResolver();
        const descriptor = resolver.getDescriptor(dtype);
        const numel = shape.reduce((acc, val) => acc * val, 1);
        let storage: WebGPUStorage;

        if (!data) {
            // 无数据时，根据物理存储需求分配空间
            const byteLength = resolver.calculateStorageBytes(dtype, numel);
            storage = new WebGPUStorage(byteLength);
        } else {
            // 有数据时，通过 Descriptor 的转换函数处理
            if (descriptor.needsUploadConversion) {
                // 需要转换：将逻辑类型转换为 WebGPU 物理存储格式
                const srcBuffer = (data as ArrayBufferView).buffer as ArrayBuffer;
                const convertedBuffer = descriptor.uploadConverter(srcBuffer, numel);
                storage = new WebGPUStorage(convertedBuffer);
            } else {
                // 无需转换：直接使用原始数据
                storage = new WebGPUStorage(data.buffer as ArrayBuffer);
            }
        }

        // 根据 memoryFormat 计算 strides
        const strides = computeStridesForFormat(shape, memoryFormat);
        return new WebGPUTensor(shape, dtype, storage, strides, 0, memoryFormat);
    }

    /**
     * 创建 View Tensor，共享已有的 Storage (零拷贝)
     * 用于 view/slice/permute/reshape 等操作
     * 
     * @param storage - 共享的 storage
     * @param shape - 视图形状
     * @param dtype - 数据类型
     * @param strides - 视图 strides
     * @param offset - 在 storage 中的偏移量
     * @param memoryFormat - 内存格式 (可选，默认从 strides 推断)
     */
    static createView<T extends DType>(
        storage: WebGPUStorage,
        shape: Shape,
        dtype: T,
        strides: number[],
        offset: number,
        memoryFormat?: MemoryFormat
    ): WebGPUTensor<T> {
        // 增加引用计数，确保 storage 不会被过早释放
        storage.incRef();

        // 如果未指定 memoryFormat，从 strides 推断
        const format = memoryFormat ?? inferMemoryFormat(shape, strides);
        return new WebGPUTensor(shape, dtype, storage, strides, offset, format);
    }

    //@ts-ignore
    get data(): DataTypeMap[T] {
        throw new Error("WebGPUTensor must call dataAsync() to get data.");
    }

    async dataAsync(): Promise<DataTypeMap[T]> {
        const resolver = getGlobalDTypeResolver();
        const descriptor = resolver.getDescriptor(this.dtype);
        const rawData = await this.storage.toRawDataAsync();

        // 通过 Descriptor 的转换函数处理下载转换
        let fullData: ArrayBuffer;
        if (descriptor.needsDownloadConversion) {
            // 需要从 WebGPU 物理格式转换回逻辑类型
            const storageNumel = rawData.byteLength / descriptor.gpuBytesPerElement;
            fullData = descriptor.downloadConverter(rawData, storageNumel);
        } else {
            fullData = rawData;
        }

        const Ctor = getTypedArrayCtor(this.dtype);
        const fullTypedArray = new Ctor(fullData);

        // 检查是否需要 strided extraction
        const isContiguous = this.offset === 0 && isContiguousStrides(this.shape, this.strides);

        // 计算物理元素数量 (complex 类型需要 x2)
        const isComplex = this.dtype === 'complex64' || this.dtype === 'complex128';
        const components = isComplex ? 2 : 1;
        const totalElements = this.numel * components;

        if (isContiguous && fullTypedArray.length === totalElements) {
            // 简单情况：连续且无 offset，返回完整数据
            return fullTypedArray.slice(0, totalElements) as DataTypeMap[T];
        }

        // 需要使用 strided extraction
        const result = new Ctor(totalElements);
        this.extractStridedData(fullTypedArray, result);

        return result as DataTypeMap[T];
    }

    /**
     * 从 strided storage 中提取数据到连续数组
     */
    private extractStridedData(source: ArrayLike<number>, dest: ArrayLike<number> & { [index: number]: number }): void {
        const ndim = this.shape.length;
        const isComplex = this.dtype === 'complex64' || this.dtype === 'complex128';
        const components = isComplex ? 2 : 1;

        if (ndim === 0) {
            // Scalar
            const srcBase = this.offset * components;
            for (let k = 0; k < components; k++) {
                dest[k] = source[srcBase + k];
            }
            return;
        }

        // 使用多维索引遍历
        const indices = new Array(ndim).fill(0);

        for (let i = 0; i < this.numel; i++) {
            // 计算源数据中的 flat index (LOGICAL)
            let srcIndex = this.offset;
            for (let d = 0; d < ndim; d++) {
                srcIndex += indices[d] * this.strides[d];
            }

            // Copy all components (1 for scalar, 2 for complex)
            const srcBase = srcIndex * components;
            const dstBase = i * components;

            for (let k = 0; k < components; k++) {
                dest[dstBase + k] = source[srcBase + k];
            }

            // 更新 indices (row-major order)
            for (let d = ndim - 1; d >= 0; d--) {
                indices[d]++;
                if (indices[d] < this.shape[d]) {
                    break;
                }
                indices[d] = 0;
            }
        }
    }


    get storageId(): number {
        return this.storage.storageId;
    }

    get device(): DeviceNameEnum {
        return this.storage.device;
    }

    /**
     * 立即释放底层 GPUBuffer
     * 
     * 警告：调用后此 Tensor 及所有共享同一 Storage 的 View 都将无效。
     * 推荐仅在确定不再需要此 Tensor 时调用。
     */
    dispose(): void {
        if (this._disposed) {
            return;
        }
        this._disposed = true;
        this.storage.dispose();
    }

    /**
     * 检查此 Tensor 是否已被释放
     */
    get isDisposed(): boolean {
        return this._disposed || this.storage.isDisposed;
    }

}

