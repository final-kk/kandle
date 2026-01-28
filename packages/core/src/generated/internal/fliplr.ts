/**
 * v5 Internal: fliplr
 * Mechanism: Shape
 * DispatchKey: fliplr
 *
 * 左右翻转 (沿 dim=1 翻转)，要求至少 2D
 *
 */

import type { ITensorHandle } from '@kandle/types';
import { opschema } from '@kandle/types';
const __entry = opschema.ops.fliplr;
import { dispatchShape, type OperatorContext } from '../../dispatch/handlers';

export function fliplr(
    self: ITensorHandle
): ITensorHandle {
    const ctx: OperatorContext = {
        opName: 'fliplr',
        tensorInputs: [self],
        scalarArgs: {},
        metadata: {},
    };
    return dispatchShape(__entry, ctx) as ITensorHandle;
}
