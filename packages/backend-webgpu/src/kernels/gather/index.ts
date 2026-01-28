/**
 * Gather Kernels Registration
 * 
 * 注册所有 Gather 操作: index_select, gather, etc.
 */

import type { IBackendOpsRegister, ITensorHandle } from '@kandle/types';
import { executeIndexSelect } from './executor';
import type { IndexSelectParams } from './types';

/**
 * 注册所有 Gather kernels
 */
export function registerGatherKernels(registry: IBackendOpsRegister): void {
    // index_select
    registry.register('index_select', (
        self: ITensorHandle,
        index: ITensorHandle,
        params: Record<string, unknown>,
        output: ITensorHandle
    ) => {
        executeIndexSelect(self, index, params as unknown as IndexSelectParams, output);
    });
}

// Re-export
export { executeIndexSelect } from './executor';
export { buildIndexSelectShader } from './shaderBuilder';
export type { IndexSelectParams, IndexSelectShaderParams } from './types';
