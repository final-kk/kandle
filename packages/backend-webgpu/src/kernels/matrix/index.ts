/**
 * Matrix Kernels Registration
 * 
 * Registers all matrix multiplication operations
 * 
 * Note: Matrix operations use specialized kernels that work directly
 * with MatmulDispatchResult configuration, not TensorIterator.
 * This is intentional as matmul requires specialized tiling algorithms.
 */

import { IBackendOpsRegister, ITensorHandle } from '@kandle/types';
import type { MatmulDispatchResult } from './types';
import { matmulExecutor } from './executor';

/**
 * Register all matrix kernels
 */
export function registerMatrixKernels(registry: IBackendOpsRegister): void {
    // MatMul: general matrix multiplication dispatcher
    registry.register('matmul', ((config: MatmulDispatchResult, a: ITensorHandle, b: ITensorHandle) => {
        matmulExecutor(config, a, b);
    }) as any);

    // MM: 2D matrix multiplication
    registry.register('mm', ((config: MatmulDispatchResult, a: ITensorHandle, b: ITensorHandle) => {
        matmulExecutor(config, a, b);
    }) as any);

    // BMM: batched matrix multiplication
    registry.register('bmm', ((config: MatmulDispatchResult, a: ITensorHandle, b: ITensorHandle) => {
        matmulExecutor(config, a, b);
    }) as any);

    // AddMM: matrix multiplication with addition (alpha * A @ B + beta * C)
    registry.register('addmm', ((config: MatmulDispatchResult, a: ITensorHandle, b: ITensorHandle) => {
        matmulExecutor(config, a, b);
    }) as any);

    // BadDBMM: batched matrix multiplication with addition
    registry.register('baddbmm', ((config: MatmulDispatchResult, a: ITensorHandle, b: ITensorHandle) => {
        matmulExecutor(config, a, b);
    }) as any);
}