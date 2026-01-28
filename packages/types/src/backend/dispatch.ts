/**
 * Dispatch Context Types
 * 
 * 定义 dispatch 层和后端交互所需的上下文类型
 * 供 kandle dispatch 层和各后端 (WebGPU/JS/Node) 使用
 */

import type { ITensorHandle, ITensorIterator } from '../tensor';
import type { OpEntry } from '../opschema';

// ============================================================================
// OperatorContext - 操作符执行上下文
// ============================================================================

/**
 * OperatorContext - 包含执行操作符所需的所有信息
 *
 * 解决 v4 的核心问题: Scalar 参数在 dispatch 过程中丢失
 */
export interface OperatorContext {
    /** 操作名称 (对应 OpEntry.dispatchKey) */
    readonly opName: string;

    /** Tensor 输入 (参与广播和迭代) */
    readonly tensorInputs: ITensorHandle[];

    /** Scalar 参数 (传给 kernel，名称 -> 值) */
    readonly scalarArgs: Record<string, number | boolean | string | number[] | undefined>;

    /** 元数据参数 (shape, axes 等非标量非 tensor) */
    readonly metadata: Record<string, unknown>;

    /**
     * 输出 Tensor 数组 (单输出时 length=1，多输出时 length>1)
     * 统一接口，替代原来的 out + extraOutputs
     */
    readonly outs?: ITensorHandle[];
}

// ============================================================================
// Execution Context - 执行上下文 (Handler 和 Kernel 共用)
// ============================================================================

/**
 * 执行上下文 - Handler 构建后透传给 kernel
 */
export type ExecutionContext =
    | IteratorContext    // 使用 TensorIterator (Pointwise, Reduction, Scan)
    | DirectContext      // 直接调用 kernel (Matrix, Factory, Triangular)
    | MetadataContext;   // 纯元数据操作 (View)

export interface IteratorContext {
    readonly kind: 'iterator';
    readonly iterator: ITensorIterator;
    readonly kernelName: string;
}

/**
 * DirectContext - 直接调用 kernel 时的上下文
 * 
 * 用于不需要 TensorIterator 的操作 (Factory, Triangular, Matrix 等)
 * Handler 构建此 context，透传给后端 kernel
 */
export interface DirectContext {
    readonly kind: 'direct';
    readonly inputs: ITensorHandle[];
    readonly scalars: Record<string, unknown>;
    /** 元数据 (如 dims 数组, 用于 Sort/Matrix 操作) */
    readonly metadata?: Record<string, unknown>;
    /** 输出 Tensor 数组 (预分配) */
    readonly outs?: ITensorHandle[];
    readonly kernelName: string;
}

export interface MetadataContext {
    readonly kind: 'metadata';
    readonly input: ITensorHandle;
    readonly params: Record<string, unknown>;
    readonly opName: string;
}

// ============================================================================
// Pattern Handler Interface
// ============================================================================

/**
 * PatternHandler - 按计算模式处理操作的抽象接口
 */
export interface PatternHandler {
    /**
     * 构建执行上下文
     */
    buildContext(entry: OpEntry, ctx: OperatorContext): ExecutionContext;

    /**
     * 执行操作
     */
    execute(execCtx: ExecutionContext): ITensorHandle | ITensorHandle[];
}

// ============================================================================
// Kernel Implementation Types
// ============================================================================

/**
 * DirectKernelImpl - 接受 DirectContext 的 kernel 实现签名
 * 
 * 使用 DirectContext 而不是单独的 DirectKernelContext，
 * 保持 Handler -> Kernel 的上下文透传一致性
 */
export type DirectKernelImpl = (ctx: DirectContext) => void;
