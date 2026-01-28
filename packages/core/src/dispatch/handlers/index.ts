/**
 * v6 Handlers Index
 *
 * 导出所有 Mechanism Handlers
 */

// Types (从 @kandle/types 重导出)
export type {
    OperatorContext,
    ExecutionContext,
    IteratorContext,
    DirectContext,
    MetadataContext,
    PatternHandler,
    DirectKernelImpl,
} from './types';
export { inferShape, inferDtype } from './types';

// v6 Generic Handlers
export { IteratorHandler, dispatchIterator } from './iterator';
export { CompositeHandler, dispatchComposite } from './composite';

// v6 Unified Shape Handler (合并了 View + Cat)
export {
    ShapeHandler,
    dispatchShape,
    dispatchView,
    dispatchCat,
} from './shape';

// v5 Handlers (Legacy / Delegated)
export { MatrixHandler, dispatchMatrix } from './matrix';
export { FactoryHandler, dispatchFactory } from './factory';
export { CopyHandler, dispatchCopy } from './copy';
export { SortHandler, dispatchSort } from './sort';
export { WindowHandler, dispatchWindow, type ConvDispatchResult } from './window';
export { NormalizeHandler, dispatchNormalize } from './normalize';
export { GatherHandler, dispatchGather } from './gather';
export { ScatterHandler, dispatchScatter } from './scatter';
export { TriangularHandler, dispatchTriangular } from './triangular';
export { WindowFuncHandler, dispatchWindowFunc } from './windowfunc';
export { FFTHandler, dispatchFFT } from './fft';
export { IIRHandler, dispatchIIR } from './iir';
