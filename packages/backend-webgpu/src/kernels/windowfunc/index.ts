/**
 * WindowFunc Kernel Module
 * 
 * Exports window function generation kernel types, executors, and registration.
 */

export * from './types';
export * from './executor';

import type { IBackendOpsRegister, DirectContext } from '@kandle/types';
import { executeWindowFunc } from './executor';
import type { WindowFuncKernelArgs } from './types';

/**
 * Register WindowFunc kernels
 */
export function registerWindowFuncKernels(registry: IBackendOpsRegister): void {
    // WindowFunc uses DirectContext pattern
    registry.register('windowfunc', (ctx: DirectContext) => {
        const args: WindowFuncKernelArgs = ctx.scalars as any;
        executeWindowFunc(args);
    });
}
