/**
 * Normalize Kernels Registration
 * 
 * 注册所有 Normalize 操作: softmax, log_softmax, layer_norm, batch_norm, etc.
 */

import type { IBackendOpsRegister, ITensorHandle } from '@kandle/types';
import { executeNormalize } from './executor';
import { NORMALIZE_OPS } from './ops';
import type { NormalizeKernelParams } from './types';

/**
 * 注册所有 Normalize kernels
 */
export function registerNormalizeKernels(registry: IBackendOpsRegister): void {
    // 注册所有 normalize 操作
    for (const dispatchKey of Object.keys(NORMALIZE_OPS)) {
        registry.register(dispatchKey, (
            inputs: ITensorHandle[],
            params: Record<string, unknown>
        ) => {
            return executeNormalize(dispatchKey, inputs, params as NormalizeKernelParams);
        });
    }
}

// Re-export
export { NORMALIZE_OPS } from './ops';
export { executeNormalize, computeNormalizedDims } from './executor';
export type { NormalizeOpConfig, NormalizeKernelParams, NormalizeShaderParams } from './types';
export { buildNormalizeShader } from './shaderBuilder';
