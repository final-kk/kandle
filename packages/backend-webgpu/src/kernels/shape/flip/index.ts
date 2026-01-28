/**
 * Flip Kernel Registration
 */

import type { IBackendOpsRegister } from '@kandle/types';
import { flipKernel } from './executor';

/**
 * Register flip kernels
 */
export function registerFlipKernels(registry: IBackendOpsRegister): void {
    registry.register('flip', flipKernel);
    // fliplr and flipud are handled in dispatch layer as flip([1]) and flip([0])
}

export * from './executor';
export * from './types';
