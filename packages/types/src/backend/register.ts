import { ITensorIterator, ITensorHandle } from "../tensor";
import { DType } from "../base";
import { DirectKernelImpl } from "./dispatch";


/**
 * Standard kernel implementation for iterator-based operations (binary, reduction, etc.)
 */
export type IteratorKernelImpl = (iter: ITensorIterator) => void;

/**
 * Copy kernel implementation for cast/contiguous operations
 */
export type CopyKernelImpl = (
    input: ITensorHandle,
    options: CopyKernelOptions
) => ITensorHandle;

export interface CopyKernelOptions {
    /** Target dtype for cast operation */
    targetDtype?: DType;
    /** Backend reference for tensor creation */
    backend: any;
}

/**
 * Sort kernel implementation for topk/sort/argsort operations
 * Uses DirectContext pattern (not TensorIterator)
 */
export type SortKernelImpl = (
    input: ITensorHandle,
    scalars: Record<string, unknown>,
    outs?: ITensorHandle[]
) => ITensorHandle | [ITensorHandle, ITensorHandle];

/**
 * Window kernel implementation for Conv/Pool operations
 * Uses DirectContext pattern with ConvDispatchResult
 */
export type WindowKernelImpl = (
    config: Record<string, unknown>,  // ConvDispatchResult
    ...inputs: ITensorHandle[]
) => ITensorHandle;

/**
 * Normalize kernel implementation for softmax/layer_norm/batch_norm operations
 * Uses DirectContext pattern
 * 
 * @param inputs - Input tensors (input, optional weight, optional bias, etc.)
 * @param params - Scalar parameters (dim, eps, normalized_shape, etc.)
 */
export type NormalizeKernelImpl = (
    inputs: ITensorHandle[],
    params: Record<string, unknown>
) => ITensorHandle;

/**
 * Gather kernel implementation for index_select/gather/embedding operations
 * Uses DirectContext pattern
 * 
 * @param self - Source tensor to select from
 * @param index - Index tensor (1D for index_select)
 * @param params - Scalar parameters (dim, etc.)
 * @param output - Pre-allocated output tensor
 */
export type GatherKernelImpl = (
    self: ITensorHandle,
    index: ITensorHandle,
    params: Record<string, unknown>,
    output: ITensorHandle
) => void;

/**
 * Scatter kernel implementation for scatter/scatter_add/scatter_reduce operations
 * Uses DirectContext pattern
 * 
 * @param self - Target tensor (initial values)
 * @param index - Index tensor
 * @param src - Source tensor
 * @param params - Scalar parameters (dim, reduce mode, etc.)
 * @param output - Pre-allocated output tensor
 */
export type ScatterKernelImpl = (
    self: ITensorHandle,
    index: ITensorHandle,
    src: ITensorHandle,
    params: Record<string, unknown>,
    output: ITensorHandle
) => void;

/**
 * FlashAttention kernel implementation for optimized scaled dot-product attention
 * Uses DirectContext pattern
 * 
 * @param query - Query tensor [batch, numHeadsQ, seqLenQ, headDim]
 * @param key - Key tensor [batch, numHeadsKV, seqLenKV, headDim]
 * @param value - Value tensor [batch, numHeadsKV, seqLenKV, headDim]
 * @param output - Pre-allocated output tensor [batch, numHeadsQ, seqLenQ, headDim]
 * @param scale - Scaling factor (typically 1/sqrt(headDim))
 * @param isCausal - Whether to apply causal masking
 */
export type FlashAttentionKernelImpl = (
    query: ITensorHandle,
    key: ITensorHandle,
    value: ITensorHandle,
    output: ITensorHandle,
    scale: number,
    isCausal: boolean,
) => void;



/**
 * Union of all kernel types (including DirectKernelImpl from dispatch.ts)
 */
export type KernelImpl =
    | IteratorKernelImpl
    | CopyKernelImpl
    | SortKernelImpl
    | WindowKernelImpl
    | NormalizeKernelImpl
    | GatherKernelImpl
    | ScatterKernelImpl
    | FlashAttentionKernelImpl

    | DirectKernelImpl;

export interface IBackendOpsRegister {
    /**
     * Register a kernel for an operation
     */
    register(opName: string, kernelFunc: KernelImpl): void;

    /**
     * Find a registered kernel
     */
    find(opName: string): KernelImpl | undefined;

    /**
     * Check if an operation is registered
     */
    has(opName: string): boolean;
}

