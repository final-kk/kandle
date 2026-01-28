import { Shape, DType, TensorData, RecursiveArray, ShapeLike } from "@kandle/types";
import { getTypedArrayCtor } from "./dtype";

export function inferShape(data: RecursiveArray, dtype: DType = 'float32'): Shape {
    const shape: number[] = [];
    let current: any = data;
    while (Array.isArray(current)) {
        shape.push(current.length);
        current = current[0];
    }

    return shape;
}

export function computeStrides(shape: Shape): number[] {
    const strides = new Array(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

export function computeNumel(shape: Shape): number {
    return shape.reduce((a: number, b: number) => a * b, 1);
}

export function computeBroadcastShape(shapeA: ShapeLike, shapeB: ShapeLike): ShapeLike {
    const ndimA = shapeA.length;
    const ndimB = shapeB.length;
    const maxDim = Math.max(ndimA, ndimB);
    const outputShape = new Array(maxDim);

    for (let i = 0; i < maxDim; i++) {
        // 从右向左对齐 (Right-alignment)
        // A 的当前维度值（如果越界则视为 1 - 也就是不存在的维度视为 1）
        const dimA = i < ndimA ? shapeA[ndimA - 1 - i] : 1;
        const dimB = i < ndimB ? shapeB[ndimB - 1 - i] : 1;

        if (dimA === dimB) {
            outputShape[maxDim - 1 - i] = dimA;
        } else if (dimA === 1) {
            outputShape[maxDim - 1 - i] = dimB;
        } else if (dimB === 1) {
            outputShape[maxDim - 1 - i] = dimA;
        } else {
            throw new Error(
                `Broadcasting error: Shapes ${JSON.stringify(shapeA)} and ${JSON.stringify(shapeB)} are incompatible.`
            );
        }
    }

    return outputShape;
}

export function computeBroadcastStrides(
    srcShape: readonly number[],
    srcStrides: readonly number[],
    targetShape: Shape
): number[] {
    const ndimSrc = srcShape.length;
    const ndimTarget = targetShape.length;

    // 目标维度不能小于源维度（广播只能增加维度或扩展维度）
    if (ndimTarget < ndimSrc) {
        throw new Error("Broadcasting error: Target shape has fewer dimensions than source shape.");
    }

    const newStrides = new Array(ndimTarget);

    for (let i = 0; i < ndimTarget; i++) {
        // 从右向左遍历
        const targetIdx = ndimTarget - 1 - i;
        const srcIdx = ndimSrc - 1 - i;

        if (srcIdx >= 0) {
            // 对应维度存在
            const sDim = srcShape[srcIdx];
            const tDim = targetShape[targetIdx];

            if (sDim === tDim) {
                newStrides[targetIdx] = srcStrides[srcIdx];
            } else if (sDim === 1) {
                // 广播发生：虽然形状变大了，但步长设为0，意味着永远读取同一个内存地址
                newStrides[targetIdx] = 0;
            } else {
                throw new Error(
                    `Broadcasting error: Dimension mismatch at offset ${i}. Source: ${sDim}, Target: ${tDim}`
                );
            }
        } else {
            // 源维度不存在（即 targetShape 在左侧扩充了维度）
            // 这种情况下，相当于在外部套了一层循环，但数据指针不移动
            newStrides[targetIdx] = 0;
        }
    }

    return newStrides;
}

export function isShapeEquals(s1: Shape, s2: Shape): boolean {
    if (s1.length !== s2.length) return false;
    for (let i = 0; i < s1.length; i++) if (s1[i] !== s2[i]) return false;
    return true;
}