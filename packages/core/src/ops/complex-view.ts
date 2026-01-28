/**
 * 复数视图操作
 * 
 * 实现 PyTorch 复数视图 API:
 * - viewAsReal: complex -> float 视图 (shape 增加 [..., 2])
 * - viewAsComplex: float -> complex 视图 (要求最后一维为 2)
 * - real: 提取实部视图
 * - imag: 提取虚部视图
 * 
 * 所有操作都是零拷贝视图操作 (共享 storage)。
 */

import type { DType, Shape, IStorage } from "@kandle/types";
import { Tensor } from '../tensor';
import { env } from '../env';
import { stack as stackOp } from '../generated/ops';
import { zeros as zerosOp } from '../generated/ops';

/**
 * 将复数张量视为实数张量
 * 
 * 零拷贝操作，返回形状为 [..., 2] 的实数张量视图，
 * 其中最后一维的 [0] 是实部，[1] 是虚部。
 * 
 * @param input complex64 或 complex128 张量
 * @returns float32 或 float64 张量视图
 * 
 * @example
 * ```ts
 * const z = new Tensor([complex(1, 2), complex(3, 4)]);  // shape: [2], complex64
 * const r = viewAsReal(z);  // shape: [2, 2], float32
 * // r.data() -> Float32Array [1, 2, 3, 4]
 * ```
 */
export function viewAsReal<T extends 'complex64' | 'complex128'>(
    input: Tensor<T>
): Tensor<T extends 'complex64' ? 'float32' : 'float64'> {
    if (input.dtype !== 'complex64' && input.dtype !== 'complex128') {
        throw new Error(`viewAsReal requires complex input, got ${input.dtype}`);
    }

    const outDtype: DType = input.dtype === 'complex64' ? 'float32' : 'float64';

    // 新的 shape: [...originalShape, 2]
    const newShape = [...input.shape, 2] as number[];

    // 新的 strides: 原始 strides 每个乘以 2，加上最后一维 stride 为 1
    const newStrides = [...input.strides.map(s => s * 2), 1];

    // offset 也需要乘以 2 (因为从 complex 元素索引变成 float 元素索引)
    const newOffset = input.offset * 2;

    // 通过后端创建 view
    const backend = input._handle.device;
    const storage = input.storage;

    // 使用 env 获取后端实例
    const backendInstance = env.getBackend(backend);

    const viewHandle = backendInstance.createTensorHandle({
        shape: newShape,
        dtype: outDtype,
        storage: storage as IStorage,
        strides: newStrides,
        offset: newOffset,
    });

    return new Tensor(viewHandle) as Tensor<T extends 'complex64' ? 'float32' : 'float64'>;
}

/**
 * 将实数张量视为复数张量
 * 
 * 零拷贝操作，要求输入的最后一维大小为 2 且是 contiguous 的。
 * 
 * @param input float32 或 float64 张量，最后一维必须为 2
 * @returns complex64 或 complex128 张量视图
 * 
 * @example
 * ```ts
 * const f = new Tensor([[1, 2], [3, 4]], { dtype: 'float32' });  // shape: [2, 2]
 * const c = viewAsComplex(f);  // shape: [2], complex64
 * // c.item(0) -> { re: 1, im: 2, __complex__: true }
 * ```
 */
export function viewAsComplex<T extends 'float32' | 'float64'>(
    input: Tensor<T>
): Tensor<T extends 'float32' ? 'complex64' : 'complex128'> {
    if (input.dtype !== 'float32' && input.dtype !== 'float64') {
        throw new Error(`viewAsComplex requires float32 or float64 input, got ${input.dtype}`);
    }

    const ndim = input.shape.length;
    if (ndim === 0) {
        throw new Error('viewAsComplex requires at least 1 dimension');
    }

    const lastDimSize = input.shape[ndim - 1];
    if (lastDimSize !== 2) {
        throw new Error(`viewAsComplex requires last dimension to be 2, got ${lastDimSize}`);
    }

    const lastStride = input.strides[ndim - 1];
    if (lastStride !== 1) {
        throw new Error('viewAsComplex requires the last dimension to be contiguous (stride=1)');
    }

    const outDtype: DType = input.dtype === 'float32' ? 'complex64' : 'complex128';

    // 新的 shape: 去掉最后一维
    const newShape = input.shape.slice(0, -1) as number[];

    // 新的 strides: 去掉最后一维，其余除以 2
    const newStrides = input.strides.slice(0, -1).map(s => s / 2);

    // offset 除以 2
    const newOffset = input.offset / 2;

    // 通过后端创建 view
    const backend = input._handle.device;
    const storage = input.storage;

    const backendInstance = env.getBackend(backend);

    const viewHandle = backendInstance.createTensorHandle({
        shape: newShape as Shape,
        dtype: outDtype,
        storage: storage as IStorage,
        strides: newStrides,
        offset: newOffset,
    });

    return new Tensor(viewHandle) as Tensor<T extends 'float32' ? 'complex64' : 'complex128'>;
}

/**
 * 获取复数张量的实部视图
 * 
 * 对于 complex 张量: 返回 stride 翻倍的 float 视图，只访问实部位置
 * 对于 real 张量: 返回自身 (同一个对象)
 * 
 * @param input 任意张量
 * @returns 实部视图 (complex) 或原张量 (real)
 * 
 * @example
 * ```ts
 * const z = new Tensor([complex(1, 2), complex(3, 4)]);  // [2], complex64
 * const re = real(z);  // [2], float32, 值 [1, 3]
 * ```
 */
export function real<T extends DType>(input: Tensor<T>): Tensor {
    if (input.dtype !== 'complex64' && input.dtype !== 'complex128') {
        // 对于实数类型，返回自身
        return input;
    }

    const outDtype: DType = input.dtype === 'complex64' ? 'float32' : 'float64';

    // 形状不变
    const newShape = [...input.shape] as number[];

    // strides 翻倍 (每隔一个 float 才是下一个复数的实部)
    const newStrides = input.strides.map(s => s * 2);

    // offset 翻倍 (从复数索引变成 float 索引)
    const newOffset = input.offset * 2;

    const backend = input._handle.device;
    const storage = input.storage;

    const backendInstance = env.getBackend(backend);

    const viewHandle = backendInstance.createTensorHandle({
        shape: newShape,
        dtype: outDtype,
        storage: storage as IStorage,
        strides: newStrides,
        offset: newOffset,
    });

    return new Tensor(viewHandle);
}

/**
 * 获取复数张量的虚部视图
 * 
 * 对于 complex 张量: 返回 stride 翻倍、offset + 1 的 float 视图
 * 对于 real 张量: 返回同形状的零张量
 * 
 * @param input 任意张量
 * @returns 虚部视图 (complex) 或零张量 (real)
 * 
 * @example
 * ```ts
 * const z = new Tensor([complex(1, 2), complex(3, 4)]);  // [2], complex64
 * const im = imag(z);  // [2], float32, 值 [2, 4]
 * ```
 */
export function imag<T extends DType>(input: Tensor<T>): Tensor {
    if (input.dtype !== 'complex64' && input.dtype !== 'complex128') {
        // 对于实数类型，返回全零张量
        return zerosOp([...input.shape], 'float32');
    }

    const outDtype: DType = input.dtype === 'complex64' ? 'float32' : 'float64';

    // 形状不变
    const newShape = [...input.shape] as number[];

    // strides 翻倍
    const newStrides = input.strides.map(s => s * 2);

    // offset 翻倍再 +1 (偏移到虚部位置)
    const newOffset = input.offset * 2 + 1;

    const backend = input._handle.device;
    const storage = input.storage;

    const backendInstance = env.getBackend(backend);

    const viewHandle = backendInstance.createTensorHandle({
        shape: newShape,
        dtype: outDtype,
        storage: storage as IStorage,
        strides: newStrides,
        offset: newOffset,
    });

    return new Tensor(viewHandle);
}

/**
 * 从实部和虚部张量构造复数张量
 * 
 * 对标 PyTorch 的 torch.complex(real, imag)。
 * 注意: 与 PyTorch 不同，这不是零拷贝操作，会创建新的存储。
 * 
 * @param realPart 实部张量 (float32 或 float64)
 * @param imagPart 虚部张量 (必须与 realPart 形状和类型相同)
 * @returns 复数张量
 * 
 * @example
 * ```ts
 * const re = new Tensor([1, 3], { dtype: 'float32' });
 * const im = new Tensor([2, 4], { dtype: 'float32' });
 * const z = torchComplex(re, im);  // shape: [2], complex64
 * // z.item(0) -> { re: 1, im: 2, __complex__: true }
 * ```
 */
export function torchComplex<T extends 'float32' | 'float64'>(
    realPart: Tensor<T>,
    imagPart: Tensor<T>
): Tensor<T extends 'float32' ? 'complex64' : 'complex128'> {
    // 类型检查
    if (realPart.dtype !== 'float32' && realPart.dtype !== 'float64') {
        throw new Error(`torchComplex requires float32 or float64 real input, got ${realPart.dtype}`);
    }
    if (imagPart.dtype !== 'float32' && imagPart.dtype !== 'float64') {
        throw new Error(`torchComplex requires float32 or float64 imag input, got ${imagPart.dtype}`);
    }
    if (realPart.dtype !== imagPart.dtype) {
        throw new Error(`dtype mismatch: real=${realPart.dtype} vs imag=${imagPart.dtype}`);
    }

    // 形状检查
    const realShape = realPart.shape;
    const imagShape = imagPart.shape;
    if (realShape.length !== imagShape.length ||
        !realShape.every((s, i) => s === imagShape[i])) {
        throw new Error(`Shape mismatch: real=[${realShape}] vs imag=[${imagShape}]`);
    }

    // 使用 stack 将两个张量沿最后一维堆叠
    // real: [2, 3] + imag: [2, 3] -> stacked: [2, 3, 2]
    // 然后 viewAsComplex 得到 [2, 3] complex
    const stacked = stackOp([realPart, imagPart], -1) as Tensor<T>;

    // 确保 stacked 是 contiguous 的（stack 结果应该已经是）
    // viewAsComplex 要求最后一维 stride = 1
    return viewAsComplex(stacked);
}
