import { ITensorHandle, RecursiveArray, ScalarDataType, TensorHandleSymbol, DType } from "@kandle/types";

export function isTensorHandle(obj: any): boolean {
    return obj && obj[TensorHandleSymbol] === true;
}

/**
 * Convert tensor to nested JS array. Only works for CPU backends with sync buffer access.
 */
export function toArray(handle: ITensorHandle): RecursiveArray | ScalarDataType {
    // Use buffer as typed array (works for JS backend)
    const buffer = handle.buffer as unknown as ArrayLike<number | bigint>;

    if (handle.numel === 0) {
        return []
    }

    if (handle.shape.length === 0) {
        const val = buffer[handle.offset];
        return handle.dtype === "bool" ? Boolean(val) : val;
    }

    const readDim = (dimIndex: number, currentOffset: number): RecursiveArray => {
        const dimSize = handle.shape[dimIndex];
        const dimStride = handle.strides[dimIndex];
        const result = [];

        if (dimIndex === handle.shape.length - 1) {
            for (let i = 0; i < dimSize; i++) {
                const realIndex = currentOffset + i * dimStride;
                result.push(buffer[realIndex]);
            }
        } else {
            for (let i = 0; i < dimSize; i++) {
                const nextOffset = currentOffset + i * dimStride;
                result.push(readDim(dimIndex + 1, nextOffset));
            }
        }
        return result as RecursiveArray;
    };

    return readDim(0, handle.offset);
}

/**
 * Synchronous tensor formatting - uses buffer directly.
 * Only works for CPU backends. For WebGPU, use formatTensorAsync.
 */
export function formatTensor(tensor: ITensorHandle): string {
    const { shape, offset, strides, dtype } = tensor;
    // Cast buffer to typed array for accessing values
    const buffer = tensor.buffer as unknown as ArrayLike<number | bigint>;

    if (tensor.numel === 0) {
        return `Tensor(\n  [],\n  shape=[${shape.join(',')}], dtype=${dtype}\n)`;
    }

    if (shape.length === 0) {
        const value = formatValue(buffer[offset], dtype);
        return `Tensor(\n  ${value},\n  shape=[], dtype=${dtype}\n)`;
    }

    const size = tensor.numel;
    const threshold = 1000;
    const edgeItems = 3;

    const getValue = (indices: number[]): string => {
        let realIndex = offset;
        for (let i = 0; i < indices.length; i++) {
            realIndex += indices[i] * strides[i];
        }
        return formatValue(buffer[realIndex], dtype);
    };

    const contentStr = formatRecursive(shape, size, threshold, edgeItems, getValue, 0, []);

    return `Tensor(\n  ${contentStr},\n  shape=[${shape.join(',')}], dtype=${dtype}\n)`;

}

/**
 * Async tensor formatting - uses dataAsync() for WebGPU and handles float16/complex properly.
 */
export async function formatTensorAsync(tensor: ITensorHandle): Promise<string> {
    const { shape, strides, dtype, offset } = tensor;

    if (tensor.numel === 0) {
        return `Tensor(\n  [],\n  shape=[${shape.join(',')}], dtype=${dtype}\n)`;
    }

    // Get data asynchronously (handles WebGPU, float16 conversion, etc.)
    const data = await tensor.dataAsync();

    if (shape.length === 0) {
        const value = formatValue(data[0], dtype);
        return `Tensor(\n  ${value},\n  shape=[], dtype=${dtype}\n)`;
    }

    const size = tensor.numel;
    const threshold = 1000;
    const edgeItems = 3;

    // For async data, the dataAsync already returns contiguous logical data
    // So we use simple indexing without strides
    const isComplex = dtype === 'complex64' || dtype === 'complex128';

    const getValue = (indices: number[]): string => {
        // Calculate flat index from shape indices
        let flatIndex = 0;
        let multiplier = 1;
        for (let i = indices.length - 1; i >= 0; i--) {
            flatIndex += indices[i] * multiplier;
            multiplier *= shape[i];
        }

        if (isComplex) {
            // Complex: read [real, imag] pair
            const realIdx = flatIndex * 2;
            const real = Number(data[realIdx]);
            const imag = Number(data[realIdx + 1]);
            return formatComplex(real, imag);
        }

        return formatValue(data[flatIndex], dtype);
    };

    const contentStr = formatRecursive(shape, size, threshold, edgeItems, getValue, 0, []);

    return `Tensor(\n  ${contentStr},\n  shape=[${shape.join(',')}], dtype=${dtype}\n)`;
}

/**
 * Format a single value for display based on dtype
 */
function formatValue(val: any, dtype: DType, physicalIndex?: number): string {
    if (dtype === 'bool') {
        return String(Boolean(val));
    }

    if (dtype === 'float16' || dtype === 'float32' || dtype === 'float64') {
        const num = Number(val);
        // Use fixed precision for floats
        if (Number.isInteger(num)) {
            return num.toFixed(1);
        }
        return num.toPrecision(6).replace(/\.?0+$/, '');
    }

    if (dtype === 'complex64' || dtype === 'complex128') {
        // For sync access, complex is interleaved in physical storage
        // This case is mainly for sync formatTensor, which may not work correctly for complex
        return String(val);
    }

    // Default: int types
    return String(val);
}

/**
 * Format a complex number as "a+bi" string
 */
function formatComplex(real: number, imag: number): string {
    const r = real.toPrecision(4).replace(/\.?0+$/, '');
    const i = Math.abs(imag).toPrecision(4).replace(/\.?0+$/, '');
    const sign = imag >= 0 ? '+' : '-';
    return `${r}${sign}${i}i`;
}

/**
 * Recursive formatter shared by sync and async versions
 */
function formatRecursive(
    shape: readonly number[],
    size: number,
    threshold: number,
    edgeItems: number,
    getValue: (indices: number[]) => string,
    currentDim: number,
    currentIndices: number[]
): string {
    const dimSize = shape[currentDim];
    const isLastDim = currentDim === shape.length - 1;

    let iterIndices: number[] = [];
    if (size > threshold && dimSize > 2 * edgeItems) {
        for (let i = 0; i < edgeItems; i++) iterIndices.push(i);
        iterIndices.push(-1); // -1 = ellipsis
        for (let i = dimSize - edgeItems; i < dimSize; i++) iterIndices.push(i);
    } else {
        for (let i = 0; i < dimSize; i++) iterIndices.push(i);
    }

    const indent = ' '.repeat(currentDim + 1);

    let result = '[';

    for (let i = 0; i < iterIndices.length; i++) {
        const idx = iterIndices[i];

        if (idx === -1) {
            result += '..., ';
            continue;
        }

        if (isLastDim) {
            const val = getValue([...currentIndices, idx]);
            result += val;
        } else {
            result += formatRecursive(shape, size, threshold, edgeItems, getValue, currentDim + 1, [...currentIndices, idx]);
        }

        if (i < iterIndices.length - 1) {
            result += ', ';
            if (!isLastDim) result += '\n' + indent;
        }
    }

    result += ']';
    return result;
}

