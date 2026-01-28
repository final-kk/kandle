/**
 * Factory Kernel Registration
 * 
 * 注册 eye, arange, repeat_interleave 等工厂操作到 WebGPU 后端
 * eye 使用 DirectContext 模式
 * arange 使用 TensorIterator 模式
 * repeat_interleave 使用 Direct kernel 模式
 */

import type { IBackendOpsRegister, DirectContext, ITensorIterator, IteratorKernelImpl, ITensorHandle } from '@kandle/types';
import { executeEye } from './executor';
import { executeArange } from './arangeExecutor';
import { repeatInterleaveKernel } from './repeatInterleaveExecutor';
import { executeOverlapAdd } from './overlapAddExecutor';
import { executeLinspace } from './linspaceExecutor';

/**
 * 注册 Factory kernels
 */
export function registerFactoryKernels(registry: IBackendOpsRegister): void {
    // Eye - DirectContext 模式
    registry.register('eye', (ctx: DirectContext) => {
        executeEye(ctx);
    });

    // Arange - TensorIterator 模式
    registry.register('arange', ((iter: ITensorIterator) => {
        executeArange(iter);
    }) as IteratorKernelImpl);

    // repeat_interleave - Direct kernel 模式
    registry.register('repeat_interleave', (
        input: ITensorHandle,
        scalars: Record<string, unknown>,
        outs?: ITensorHandle[]
    ) => {
        return repeatInterleaveKernel(input, scalars, outs);
    });

    // overlap_add - iSTFT 重建用
    registry.register('overlap_add', (ctx: DirectContext) => {
        executeOverlapAdd(ctx);
    });

    // linspace
    registry.register('linspace', ((iter: ITensorIterator) => {
        executeLinspace(iter);
    }) as IteratorKernelImpl);
}
