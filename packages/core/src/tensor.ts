import { DType, ITensorHandle, DataTypeMap, TensorData, TensorDataLike, ScalarDataType, RecursiveArray, DeviceNameEnum, IStorage, StructureDataType, TensorOptions, DEFAULT_FLOAT_DTYPE, MemoryFormat, Complex } from "@kandle/types";
import type { Shape } from "@kandle/types";
import { env } from './env';
import { getTypedArrayCtor, inferShape, isTypedArray, isTensorHandle, toArray, formatTensor, isContiguousStrides, isComplex, flattenComplexArray, findFirstElement } from "@kandle/utils";
import { contiguous as internalContiguous } from './generated/internal/contiguous';
import { _trackTensor } from './memory';

export class Tensor<T extends DType = DType> {

    /**
     * @internal
     * @description The backend tensor handle associated with this Tensor instance.
     */
    _handle: ITensorHandle;

    /**
     * @internal
     * Scope ID - 标识此张量属于哪个 tidy scope
     * 用于在 endScope 时正确追踪和释放张量
     */
    scopeId?: number;

    /**
     * @internal
     * 是否被 keep() 标记为保留，不会被 tidy 自动释放
     */
    kept?: boolean;

    /**
     * @description
     * Create a new Tensor.
     *
     * Examples:
     * ```ts
     * // 1. From a flat array (default float32)
     * new Tensor([1, 2, 3, 4]); // shape: [4], dtype: float32
     *
     * // 2. From a flat array with options
     * new Tensor([1, 2, 3, 4], { dtype: 'int32' }); // shape: [4], dtype: int32
     *
     * // 3. From a nested array (shape inferred)
     * new Tensor([[1, 2], [3, 4]]); // shape: [2, 2], dtype: float32
     *
     * // 4. From a TypedArray with explicit shape
     * new Tensor(new Float32Array([1, 2, 3, 4]), { shape: [2, 2] }); // shape: [2, 2], dtype: float32
     *
     * // 5. Scalar
     * new Tensor(5); // shape: [], dtype: float32
     * ```
     */
    constructor(handle: ITensorHandle);
    constructor(data: TensorDataLike, options?: TensorOptions);
    constructor(data: TensorDataLike, dtype?: T, device?: DeviceNameEnum); // Shortcut for: new Tensor(data, { dtype, device })

    constructor(
        arg0: TensorDataLike | ITensorHandle,
        arg1?: TensorOptions | T,
        arg2?: DeviceNameEnum
    ) {

        // 1. raw handle wrapping
        if (isTensorHandle(arg0)) {
            this._handle = arg0 as ITensorHandle;
            _trackTensor(this);  // 追踪新创建的张量
            return;
        }

        // 2. Param Parsing
        const data: TensorDataLike = arg0 as TensorDataLike;
        let dtype: DType | undefined;
        let shape: Shape | undefined;
        let device: DeviceNameEnum | undefined;

        if (typeof arg1 === 'string') {
            // new Tensor(data, dtype, device)
            dtype = arg1 as DType;
            device = arg2;
        } else if (typeof arg1 === 'object' && arg1 !== null && !Array.isArray(arg1)) {
            // new Tensor(data, options)
            const options = arg1 as TensorOptions;
            if (options.dtype) dtype = options.dtype;
            if (options.device) device = options.device;
            if (options.shape) shape = options.shape;
        }
        // Note: We intentionally removed the new Tensor(data, shape) overload to avoid ambiguity.
        // Users should use new Tensor(data, { shape: ... }) instead.

        let _data: TensorData;
        let _shape: Shape;

        if (typeof data === 'number' || typeof data === 'bigint' || typeof data === 'boolean') {
            // Case A: Scalar
            if (shape) {
                throw new Error("Cannot specify 'shape' when constructing a Tensor from a scalar.");
            }
            if (!dtype) dtype = 'float32'; // Default

            // If data is scalar, we trust it matches the requested dtype (or JS executes type coercion)
            const ctor = getTypedArrayCtor(dtype);
            _data = new ctor(1);
            (_data as any)[0] = data; // Coercion happens here naturally by standard TypedArray behavior
            _shape = [];
        } else if (isComplex(data)) {
            // Case A': Complex scalar
            if (shape) {
                throw new Error("Cannot specify 'shape' when constructing a Tensor from a complex scalar.");
            }
            // 默认 complex64，用户可通过 options.dtype 指定 complex128
            if (!dtype) dtype = 'complex64';
            if (dtype !== 'complex64' && dtype !== 'complex128') {
                throw new Error(`Cannot create ${dtype} tensor from Complex. Use complex64 or complex128.`);
            }

            const ArrayCtor = dtype === 'complex128' ? Float64Array : Float32Array;
            _data = new ArrayCtor([data.re, data.im]);
            _shape = []; // scalar
        } else if (isTypedArray(data)) {
            // Case B: TypedArray
            if (!dtype) {
                // Infer dtype from TypedArray
                if (data instanceof Float32Array) dtype = 'float32';
                else if (data instanceof Int32Array) dtype = 'int32';
                else if (data instanceof Uint8Array) dtype = 'bool'; // Assuming uint8 is used for bool
                else dtype = 'float32'; // Default fallback
            }

            const targetCtor = getTypedArrayCtor(dtype);
            if (data.constructor === targetCtor) {
                _data = data as TensorData;
            } else {
                // Type mismatch (e.g. Int32Array passed but float32 requested), perform copy/cast
                _data = new targetCtor(data);
            }
            // If the user did not provide a shape, this becomes a one-dimensional vector
            _shape = shape || [_data.length];
        } else if (Array.isArray(data)) {
            // Case C: Recursive Array  e.g. [[1,2], [3,4]] or [complex(1,2), complex(3,4)]
            const firstElement = findFirstElement(data);

            if (isComplex(firstElement)) {
                // Case C': Complex array - 必须是纯 Complex 数组
                if (!dtype) dtype = 'complex64';
                if (dtype !== 'complex64' && dtype !== 'complex128') {
                    throw new Error(`Cannot create ${dtype} tensor from Complex array. Use complex64 or complex128.`);
                }

                const { flat, shape: inferredShape } = flattenComplexArray(data, dtype as 'complex64' | 'complex128');
                _data = flat;
                _shape = shape || inferredShape;
            } else {
                // Case C: Regular recursive array
                if (!dtype) dtype = DEFAULT_FLOAT_DTYPE;

                _shape = shape || inferShape(data, dtype);
                const flatData = (data as any[]).flat(Infinity);

                // Special handling for float16: user provides logical float values
                // Backend will convert Float32Array → Uint16Array (physical storage)
                if (dtype === 'float16') {
                    _data = new Float32Array(flatData);
                } else {
                    const Ctor = getTypedArrayCtor(dtype);
                    _data = new Ctor(flatData);
                }
            }
        } else {
            throw new Error("Unsupported data type for Tensor construction.");
        }

        // 计算期望的数据长度
        // 对于 complex 类型，每个元素需要 2 个基础类型值 (real, imag)
        const size = _shape.reduce((a, b) => a * b, 1);
        const isComplexDtype = dtype === 'complex64' || dtype === 'complex128';
        const expectedDataLength = isComplexDtype ? size * 2 : size;

        if (_data.length !== expectedDataLength) {
            if (isComplexDtype) {
                throw new Error(`Shape ${_shape} (size ${size}) requires ${expectedDataLength} values for ${dtype} (2 per complex), but got ${_data.length}`);
            } else {
                throw new Error(`Shape ${_shape} (size ${size}) does not match data length ${_data.length}`);
            }
        }

        const targetDevice = device ? env.getBackend(device) : env.getDefaultDevice();
        // console.log(`Creating Tensor on device: ${targetDevice.name}`);
        this._handle = targetDevice.createTensorHandle(_shape, dtype as T, _data);
        _trackTensor(this);  // 追踪新创建的张量
    }

    get dims(): Shape {
        return this._handle.shape;
    }

    get shape(): Shape {
        return this._handle.shape;
    }

    get dtype(): T {
        return this._handle.dtype as T;
    }

    // get data(): DataTypeMap[T] {
    //     return this._handle.data;
    // }

    get numel(): number {
        return this._handle.numel;
    }

    get size(): number {
        return this._handle.numel;
    }

    get strides(): readonly number[] {
        return this._handle.strides;
    }

    get offset(): number {
        return this._handle.offset;
    }

    get id(): number {
        return this._handle.id;
    }

    get storage(): IStorage {
        return this._handle.storage;
    }

    get storageId(): number {
        return this._handle.storageId;
    }

    get backend(): DeviceNameEnum {
        return this._handle.device;
    }

    get device(): DeviceNameEnum {
        return this._handle.device;
    }

    /**
     * @description Synchronously retrieves the tensor data as a TypedArray.
     *
     * Note: In web environments, this may not be available for GPU-backed tensors.
     */
    get data(): DataTypeMap[T] {
        const Ctor = getTypedArrayCtor(this.dtype);
        return new Ctor(this._handle.buffer as ArrayBuffer);
    }

    /**
     * @description in web environments, there is no sync way to get data from GPUBuffer or other async storages
     *
     * so we provide an async method to get the data.
     */
    async dataAsync(): Promise<DataTypeMap[T]> {
        // 委托给 handle 的 dataAsync，这样后端可以执行必要的类型转换
        // (例如 WebGPU 的 float64 从 f32 转换回来)
        return await this._handle.dataAsync() as DataTypeMap[T];
    }

    /*----------------- ONNX Runtime  -----------------*/
    /**
     * @description Returns arguments suitable for the onnxruntime Tensor constructor.
     * Usage:
     * ```ts
    * import * as Ort from 'onnxruntime';
     * new Ort.Tensor(...tensor.toOrtArgs())
    * ```
     */
    toOrtArgs(): [DType, TensorData, Shape] {
        return [this.dtype, this.data, this.shape as number[]];
    }

    static fromOrtTensor() { }

    /*----------------- js side utils -----------------*/
    /**
     * @description If the Tensor has a single element,
     * returns that element as a standard JavaScript number, bigint, boolean, or Complex.
     * For complex tensors, returns a Complex object { re, im, __complex__: true }.
     */
    item(): ScalarDataType | Complex {
        if (this.numel !== 1) {
            throw new Error(`Cannot convert a Tensor with ${this.numel} elements to a scalar.`);
        }

        if (this.dtype === 'complex64' || this.dtype === 'complex128') {
            const data = this.data as any;
            return {
                re: data[0],
                im: data[1],
                __complex__: true as const,
            };
        }

        const value = this.data[0];
        return this.dtype === "bool" ? Boolean(value) : value;
    }

    toArray(): RecursiveArray | ScalarDataType {
        return toArray(this._handle);
    }

    toString(): string {
        return formatTensor(this._handle);
    }

    [Symbol.for('nodejs.util.inspect.custom')](depth: number, options: any): string {
        return this.toString();
    }

    isSharedWith(other: Tensor<any>): boolean {
        return this.storageId === other.storageId;
    }

    /**
     * Returns true if the tensor data is stored contiguously in memory.
     * A tensor is contiguous if its strides match the expected row-major layout.
     */
    get isContiguous(): boolean {
        return isContiguousStrides(this.shape, this._handle.strides);
    }

    /**
     * 获取当前内存格式
     * 
     * - Contiguous: 标准 row-major (NCHW for 4D)
     * - ChannelsLast: 通道维度最密集 (NHWC physical layout)
     * - ChannelsLast3d: 5D 版本 (NDHWC)
     */
    get memoryFormat(): MemoryFormat {
        return this._handle.memoryFormat;
    }

    /**
     * 检查是否是 channels_last 格式 (包括 4D 和 5D)
     */
    get isChannelsLast(): boolean {
        return this._handle.memoryFormat === MemoryFormat.ChannelsLast ||
            this._handle.memoryFormat === MemoryFormat.ChannelsLast3d;
    }

    /**
     * 转换到指定内存格式
     * 
     * 对标 PyTorch 的 `tensor.to(memory_format =...)` 或 `tensor.contiguous(memory_format =...)`。
     * 
     * @param format 目标内存格式:
     *   - `MemoryFormat.Contiguous`: 标准 row-major (NCHW for 4D)
     *   - `MemoryFormat.ChannelsLast`: 通道最密集 (NHWC physical layout for 4D)
     *   - `MemoryFormat.ChannelsLast3d`: 5D 版本 (NDHWC)
     *   - `MemoryFormat.Preserve`: 保持当前格式 (返回原 tensor)
     * @returns 指定格式的 Tensor，如果已是目标格式则返回 this (zero-copy)
     * 
     * @example
     * ```ts
    * // 转换为 channels_last (NHWC) 格式
     * const nhwc = tensor.toMemoryFormat(MemoryFormat.ChannelsLast);
     * 
     * // 转换回标准格式
     * const nchw = nhwc.toMemoryFormat(MemoryFormat.Contiguous);
     * ```
     */
    toMemoryFormat(format: MemoryFormat): Tensor {
        // Preserve 直接返回 this
        if (format === MemoryFormat.Preserve) {
            return this;
        }
        // 如果已是目标格式，返回 this
        if (this._handle.memoryFormat === format) {
            return this;
        }
        // 调用 internal contiguous 函数
        return new Tensor(internalContiguous(this._handle, format));
    }

    /**
     * 立即释放底层 GPU 内存
     * 
     * 在撤销 Memory Pool 架构后，GPU 内存依赖 JS GC 释放，
     * 但 GC 时机不可控可能导致显存积压。在需要主动管理内存时调用此方法。
     * 
     * 警告：调用后此 Tensor 及所有共享同一 Storage 的 View 都将无效。
     * 任何后续访问（dataAsync、作为算子输入等）都会导致未定义行为。
     * 
     * @example
     * ```ts
    * // 显式释放不再需要的中间结果
     * const intermediate = matmul(a, b);
     * const result = relu(intermediate);
     * intermediate.dispose(); // 立即释放显存
     * ```
     */
    dispose(): void {
        if (!this.isDisposed) {
            if ('dispose' in this._handle) {
                (this._handle as any).dispose();
            }
        }
    }

    /**
     * 检查此 Tensor 是否已被释放
     */
    get isDisposed(): boolean {
        if ('isDisposed' in this._handle) {
            return this._handle.isDisposed as boolean;
        }
        return false;
    }

}
