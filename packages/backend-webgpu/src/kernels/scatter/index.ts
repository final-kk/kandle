/**
 * Scatter Kernels Registration
 *
 * 注册所有 Scatter 操作: scatter, scatter_add, scatter_reduce
 */

import type { IBackendOpsRegister, ITensorHandle } from '@kandle/types';
import { executeScatter, executeScatterAdd, executeScatterReduce } from './executor';
import type { ScatterParams, ScatterAddParams, ScatterReduceParams, ScatterReduceMode } from './types';

/**
 * 注册所有 Scatter kernels
 */
export function registerScatterKernels(registry: IBackendOpsRegister): void {
    // scatter
    registry.register('scatter', (
        self: ITensorHandle,
        index: ITensorHandle,
        src: ITensorHandle,
        params: Record<string, unknown>,
        output: ITensorHandle
    ) => {
        executeScatter(self, index, src, params as unknown as ScatterParams, output);
    });

    // scatter_add
    registry.register('scatter_add', (
        self: ITensorHandle,
        index: ITensorHandle,
        src: ITensorHandle,
        params: Record<string, unknown>,
        output: ITensorHandle
    ) => {
        executeScatterAdd(self, index, src, params as unknown as ScatterAddParams, output);
    });

    // scatter_reduce
    registry.register('scatter_reduce', (
        self: ITensorHandle,
        index: ITensorHandle,
        src: ITensorHandle,
        params: Record<string, unknown>,
        output: ITensorHandle
    ) => {
        const scatterParams: ScatterReduceParams = {
            dim: params['dim'] as number,
            reduce: params['reduce'] as ScatterReduceMode,
            includeSelf: params['includeSelf'] as boolean ?? true,
        };
        executeScatterReduce(self, index, src, scatterParams, output);
    });
}

// Re-export
export { executeScatter, executeScatterAdd, executeScatterReduce } from './executor';
export { buildScatterShader, buildScatterAddShader, buildScatterReduceShader } from './shaderBuilder';
export type {
    ScatterParams,
    ScatterAddParams,
    ScatterReduceParams,
    ScatterOpConfig,
    ScatterShaderParams,
    ScatterKernelImpl,
    ScatterReduceMode,
} from './types';
