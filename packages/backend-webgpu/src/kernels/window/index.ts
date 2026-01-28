/**
 * Window Kernels Registration
 * 
 * 注册所有 Conv/Pool 操作到 WebGPU 后端
 * 
 * @module kernels/window
 */

import type { IBackendOpsRegister, KernelImpl } from '@kandle/types';
import { windowExecutor } from './executor';

// Re-export types
export type {
    ConvDispatchResult,
    ConvVariant,
    PoolVariant,
    ConvAlgorithm,
    Im2ColConfig,
    ConvGemmConfig,
    PoolKernelConfig,
} from './types';

// Re-export executor
export { windowExecutor };

// Cast to KernelImpl for type compatibility
const windowKernel = windowExecutor as unknown as KernelImpl;

/**
 * 注册所有 Window (Conv/Pool) kernels
 */
export function registerWindowKernels(registry: IBackendOpsRegister): void {
    // ========================================
    // Convolution Operations
    // ========================================

    // Conv1d
    registry.register('conv1d', windowKernel);

    // Conv2d
    registry.register('conv2d', windowKernel);

    // Conv3d
    registry.register('conv3d', windowKernel);

    // ConvTranspose2d
    registry.register('conv_transpose2d', windowKernel);

    // ========================================
    // Pooling Operations
    // ========================================

    // MaxPool
    registry.register('max_pool1d', windowKernel);
    registry.register('max_pool2d', windowKernel);
    registry.register('max_pool3d', windowKernel);

    // AvgPool
    registry.register('avg_pool1d', windowKernel);
    registry.register('avg_pool2d', windowKernel);
    registry.register('avg_pool3d', windowKernel);

    // AdaptivePool
    registry.register('adaptive_avg_pool2d', windowKernel);
    registry.register('adaptive_max_pool2d', windowKernel);
}
