/**
 * Parameter - 可学习参数
 *
 * 对标 PyTorch nn.Parameter，本质是一个标记为 "需要梯度" 的 Tensor
 * 在当前纯推理阶段，Parameter 主要用于:
 * 1. 区分 "可学习权重" 和 "普通 Tensor"
 * 2. 自动注册到 Module 的 _parameters 中
 * 3. 支持 state_dict 序列化
 *
 * @see https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html
 */

import type { DType, ITensorHandle, TensorDataLike, TensorOptions } from '@kandle/types';
import { Tensor } from '../tensor';

/**
 * Parameter 类 - 可学习参数
 *
 * 类似于 Tensor，但会被 Module 自动识别为参数
 */
export class Parameter<T extends DType = DType> extends Tensor<T> {
    /**
     * 是否需要梯度 (用于未来反向传播)
     * 当前推理阶段仅作标记用
     */
    requiresGrad: boolean;

    /**
     * 梯度张量 (预留给未来反向传播使用)
     */
    grad: Tensor | null = null;

    /**
     * 从现有 Tensor 创建 Parameter
     */
    constructor(data: Tensor<T>, requiresGrad?: boolean);
    /**
     * 从 TensorHandle 创建 Parameter
     */
    constructor(handle: ITensorHandle, requiresGrad?: boolean);
    /**
     * 从数据创建 Parameter
     */
    constructor(data: TensorDataLike, options?: TensorOptions & { requiresGrad?: boolean });
    constructor(data: TensorDataLike, dtype?: T, requiresGrad?: boolean);

    constructor(
        arg0: Tensor<T> | ITensorHandle | TensorDataLike,
        arg1?: boolean | TensorOptions & { requiresGrad?: boolean } | T,
        arg2?: boolean
    ) {
        // 默认 requiresGrad = true
        let requiresGrad = true;

        if (arg0 instanceof Tensor) {
            // 从 Tensor 创建 - 共享 handle
            super(arg0._handle);
            if (typeof arg1 === 'boolean') {
                requiresGrad = arg1;
            }
        } else if (isTensorHandle(arg0)) {
            // 从 handle 创建
            super(arg0 as ITensorHandle);
            if (typeof arg1 === 'boolean') {
                requiresGrad = arg1;
            }
        } else {
            // 从数据创建
            if (typeof arg1 === 'object' && arg1 !== null && !Array.isArray(arg1)) {
                // options 形式
                const { requiresGrad: rg, ...tensorOptions } = arg1 as TensorOptions & { requiresGrad?: boolean };
                super(arg0 as TensorDataLike, tensorOptions);
                if (rg !== undefined) {
                    requiresGrad = rg;
                }
            } else if (typeof arg1 === 'string') {
                // dtype 形式
                super(arg0 as TensorDataLike, arg1 as T);
                if (typeof arg2 === 'boolean') {
                    requiresGrad = arg2;
                }
            } else {
                super(arg0 as TensorDataLike);
                if (typeof arg1 === 'boolean') {
                    requiresGrad = arg1;
                }
            }
        }

        this.requiresGrad = requiresGrad;
    }

    /**
     * 克隆为普通 Tensor (分离梯度计算图)
     *
     * 返回一个共享数据但不参与梯度计算的 Tensor
     */
    detach(): Tensor<T> {
        return new Tensor(this._handle);
    }

    /**
     * 创建一个新的 Parameter，数据相同但可能有不同的 requiresGrad
     */
    clone(requiresGrad?: boolean): Parameter<T> {
        // TODO: 实现真正的数据复制 (需要 clone op)
        // 当前仅创建新包装
        return new Parameter(this._handle, requiresGrad ?? this.requiresGrad);
    }
}

/**
 * 类型守卫 - 检查对象是否为 Parameter
 */
export function isParameter(obj: unknown): obj is Parameter {
    return obj instanceof Parameter;
}

/**
 * 辅助函数 - 检查是否为 TensorHandle
 */
function isTensorHandle(obj: unknown): obj is ITensorHandle {
    return (
        typeof obj === 'object' &&
        obj !== null &&
        'shape' in obj &&
        'dtype' in obj &&
        'storage' in obj
    );
}
