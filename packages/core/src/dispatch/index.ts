/**
 * Dispatch Module v6
 *
 * 导出所有 dispatcher 相关功能
 * 
 * ## v6 Architecture
 * - **IteratorHandler**: 处理 Iterator mechanism (Map, Reduce, Scan)
 * - **CompositeHandler**: 处理 Composite mechanism (纯 JS 组合操作)
 * - **专用 Handlers**: Matrix, Window, Normalize, Sort, Gather, Scatter, Triangular, Shape
 * - **ViewHandler, FactoryHandler, CopyHandler**: 直接处理对应 mechanism
 */

// Types
export type {
    OperatorContext,
    ExecutionContext,
    IteratorContext,
    DirectContext,
    MetadataContext,
    PatternHandler,
} from './handlers';
export { inferShape, inferDtype } from './handlers';

// v6 Mechanism Handlers (Primary)
export {
    IteratorHandler,
    dispatchIterator,
    CompositeHandler,
    dispatchComposite,
} from './handlers';

// v6 Direct Mechanism Handlers
export {
    ShapeHandler,
    dispatchShape,
    dispatchView,
    dispatchCat,
    FactoryHandler,
    dispatchFactory,
    CopyHandler,
    dispatchCopy,
} from './handlers';

// v6 Specialized Handlers (delegated from KernelHandler)
export {
    MatrixHandler,
    dispatchMatrix,
    NormalizeHandler,
    dispatchNormalize,
    SortHandler,
    dispatchSort,
    WindowHandler,
    dispatchWindow,
    GatherHandler,
    dispatchGather,
    ScatterHandler,
    dispatchScatter,
    TriangularHandler,
    dispatchTriangular,
    WindowFuncHandler,
    dispatchWindowFunc,
    FFTHandler,
    dispatchFFT,
    IIRHandler,
    dispatchIIR,
} from './handlers';

// TensorIterator
export { TensorIterator } from './TensorIterator';

// Matmul (specialized dispatch, not using handlers)
export { dispatchMatmul, dispatchMmStrict, dispatchBmmStrict } from './matmulOps';

