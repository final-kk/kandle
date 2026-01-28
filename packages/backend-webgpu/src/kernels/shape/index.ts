/**
 * Shape Operations Registration
 */

import type { IBackendOpsRegister } from '@kandle/types';
import { registerDiffKernels } from './diff';
import { registerFlipKernels } from './flip';

/**
 * Register shape kernels
 */
export function registerShapeKernels(registry: IBackendOpsRegister): void {
    registerDiffKernels(registry);
    registerFlipKernels(registry);
}

export * from './diff';
export * from './flip';

