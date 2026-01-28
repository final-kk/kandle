/**
 * NN-Kit I/O 模块
 * 
 * 统一导出所有 I/O 相关功能:
 * - ByteSource: 平台抽象层
 * - Safetensor: Safetensors 格式解析
 * - NPY: NumPy 格式解析
 * - Tensor Loaders: 从描述符创建 Tensor
 */

// ============================================================================
// Layer 0: ByteSource
// ============================================================================

export type {
    ByteSource,
    ResolvableByteSource,
} from './source';

export {
    WebByteSource,
    ArrayBufferByteSource,
    FileByteSource,
    createByteSource,
    createResolvableByteSource,
} from './source';

// ============================================================================
// Layer 1: Safetensors
// ============================================================================
export type {
    // Types
    SafetensorsDType,
    SafetensorLayer,
    SafetensorFile,
    SafetensorGroup,
    SafetensorsHeader,
    SafetensorsHeaderEntry,
    SafetensorsIndexJson,
    SAFETENSORS_DTYPE_MAP,
    SAFETENSORS_DTYPE_BYTES,
} from './safetensor';

export {

    // DType utilities
    mapSafetensorsDType,
    convertBF16toF32,
    createTypedArrayFromBuffer,
    getDtypeBytes,
    // Parsing
    parseHeaderSize,
    parseJsonHeader,
    parseSafetensorFile,
    createSafetensorGroup,
    createSingleFileGroup,
    parseShardedIndex,
    // Public API
    loadSafetensor,
} from './safetensor';

// ============================================================================
// Layer 1: NPY
// ============================================================================

export type {
    NpyDescriptor,
} from './npy';

export {
    parseNpy,
    loadNpy,
} from './npy';

// ============================================================================
// Layer 2: Tensor Loaders
// ============================================================================

export type {
    TensorLoadOptions,
} from './tensor-loaders';

export {
    tensorFromSafetensorLayer,
    tensorFromNpy,
} from './tensor-loaders';
