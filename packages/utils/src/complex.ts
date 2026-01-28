/**
 * 复数工具函数
 * 
 * 在整个 JS 环境中处理 Complex 类型时使用
 */

import type { Complex } from "@kandle/types";

/**
 * 创建复数 (默认 complex64 精度)
 * 
 * @param re 实部
 * @param im 虚部 (默认 0)
 * @returns Complex 对象
 * 
 * @example
 * ```ts
 * const z = complex(3, 4);  // 3 + 4i
 * const r = complex(5);     // 5 + 0i (纯实数)
 * ```
 */
export function complex(re: number, im: number = 0): Complex {
    return { re, im, __complex__: true as const };
}

/**
 * 检测值是否为 Complex 对象
 * 
 * @param value 待检测的值
 * @returns value 是否为 Complex
 */
export function isComplex(value: unknown): value is Complex {
    return (
        typeof value === 'object' &&
        value !== null &&
        '__complex__' in value &&
        (value as Complex).__complex__ === true
    );
}

/**
 * 从 Complex 数组扁平化为 Float32Array (complex64) 或 Float64Array (complex128)
 * 用于 Tensor 构造时的内部数据转换
 * 
 * @param data Complex 的嵌套数组
 * @param precision 精度: 'complex64' | 'complex128'
 * @returns { flat: TypedArray, shape: number[] }
 */
export function flattenComplexArray(data: unknown, precision: 'complex64' | 'complex128' = 'complex64'): {
    flat: Float32Array | Float64Array;
    shape: number[]
} {
    const shape: number[] = [];

    // 推断 shape
    let current: unknown = data;
    while (Array.isArray(current)) {
        shape.push(current.length);
        current = current[0];
    }

    // 计算元素总数
    const size = shape.reduce((a, b) => a * b, 1);

    // 分配交错存储数组
    const ArrayCtor = precision === 'complex128' ? Float64Array : Float32Array;
    const flat = new ArrayCtor(size * 2);

    // 递归填充
    let idx = 0;
    function walk(arr: unknown): void {
        if (isComplex(arr)) {
            flat[idx++] = arr.re;
            flat[idx++] = arr.im;
        } else if (Array.isArray(arr)) {
            for (const item of arr) {
                walk(item);
            }
        } else {
            throw new Error(`Invalid element in complex array: expected Complex, got ${typeof arr}`);
        }
    }
    walk(data);

    return { flat, shape };
}

/**
 * 查找嵌套数组中的第一个标量元素
 * 用于类型检测
 */
export function findFirstElement(data: unknown): unknown {
    if (Array.isArray(data)) {
        return findFirstElement(data[0]);
    }
    return data;
}
