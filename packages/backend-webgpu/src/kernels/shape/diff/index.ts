/**
 * Diff Kernels Registration
 * 
 * Registers diff operation for N-order forward difference
 */

import type { IBackendOpsRegister, ITensorHandle } from '@kandle/types';
import { diffKernel } from './executor';

/**
 * Register diff kernel
 */
export function registerDiffKernels(registry: IBackendOpsRegister): void {
    // diff: N-order forward difference
    registry.register('diff', (
        inputs: ITensorHandle[],
        params: Record<string, unknown>,
        outs?: ITensorHandle[]
    ) => {
        const [input] = inputs;
        return diffKernel(input, params, outs);
    });
}

export { diffKernel } from './executor';
export { buildDiffShader } from './shaderBuilder';
export type { DiffUniforms, DiffParams } from './types';
