/**
 * Random Kernels Registration
 * 
 * Registers rand, randn, randint, multinomial kernels to the WebGPU backend
 */

import type { IBackendOpsRegister, ITensorIterator, ITensorHandle } from '@kandle/types';
import { executeRandom } from './executor';
import { multinomialKernel } from './multinomialExecutor';
import type { RandomOpType } from './types';

/**
 * Random operations to register
 */
const RANDOM_OPS: RandomOpType[] = ['rand', 'randn', 'randint'];

/**
 * Register all random kernels
 */
export function registerRandomKernels(registry: IBackendOpsRegister): void {
    // Standard random operations (using TensorIterator)
    for (const opType of RANDOM_OPS) {
        registry.register(opType, (iter: ITensorIterator) => {
            executeRandom(iter, opType);
        });
    }

    // Multinomial (using Direct kernel pattern)
    registry.register('multinomial', (
        input: ITensorHandle,
        scalars: Record<string, unknown>,
        outs?: ITensorHandle[]
    ) => {
        return multinomialKernel(input, scalars, outs);
    });
}
