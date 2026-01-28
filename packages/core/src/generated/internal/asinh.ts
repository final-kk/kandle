/**
 * v5 Internal: asinh
 * Mechanism: Iterator
 * DispatchKey: asinh
 *
 * 逐元素反双曲正弦: arcsinh(self)
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.asinh;
import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';

export function asinh(
    self: ITensorHandle,
    out?: ITensorHandle | undefined
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'asinh',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
        ...(out !== undefined ? { outs: [out] } : {}),
    };
    return dispatchIterator(__entry, ctx) as ITensorHandle;
}
