/**
 * v5 Internal: flipud
 * Mechanism: Shape
 * DispatchKey: flipud
 *
 * 上下翻转 (沿 dim=0 翻转)，要求至少 1D
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.flipud;
import { dispatchShape, type OperatorContext } from '../../dispatch/handlers';

export function flipud(
    self: ITensorHandle
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'flipud',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
    };
    return dispatchShape(__entry, ctx) as ITensorHandle;
}
