/**
 * Scatter Kernels - WGSL Shader Builder
 *
 * 生成 scatter, scatter_add, scatter_reduce 的 WGSL shader
 *
 * 实现要点:
 * 1. scatter: 直接写入，多索引冲突时不确定
 * 2. scatter_add: 使用 atomicCompareExchangeWeak CAS loop 实现浮点原子加
 * 3. scatter_reduce: 使用 CAS loop 实现各种归约操作
 * 
 * 工业级实现：原生支持 strided (非连续) 输入
 * 通过 shape/strides/offset 计算物理地址，无需预先克隆
 */

import type { DType } from '@kandle/types';
import type { ScatterShaderParams, ScatterReduceMode } from './types';
import { getReduceIdentity, getReduceOp } from './ops';

/**
 * 获取 WGSL 数据类型
 */
function wgslType(dtype: DType): string {
    switch (dtype) {
        case 'float32': return 'f32';
        case 'float64': return 'f32';  // 退化为 f32
        case 'int32': return 'i32';
        case 'int64': return 'i32';    // 退化为 i32
        case 'uint32': return 'u32';
        case 'int16': return 'i32';
        case 'uint16': return 'u32';
        case 'int8': return 'i32';
        case 'uint8': return 'u32';
        case 'bool': return 'u32';
        default: return 'f32';
    }
}

/**
 * 判断是否为浮点类型
 */
function isFloatType(dtype: DType): boolean {
    return dtype === 'float32' || dtype === 'float64';
}

/**
 * 判断是否为有符号整型
 */
function isSignedIntType(dtype: DType): boolean {
    return dtype === 'int32' || dtype === 'int64' || dtype === 'int16' || dtype === 'int8';
}

/**
 * 生成 strided 坐标计算辅助函数 (工业级实现)
 * 
 * 支持非连续内存布局：使用 strides 和 offset 计算物理地址
 */
function generateStridedCoordFunctions(ndim: number): string {
    return `
// 将线性索引转换为 N 维坐标 (用于逻辑坐标展开)
fn unflatten_logical(idx: u32, shape: ptr<function, array<u32, ${ndim}>>) -> array<u32, ${ndim}> {
    var coords: array<u32, ${ndim}>;
    var remaining = idx;
    for (var i = 0u; i < NDIM; i++) {
        // 计算后缀维度乘积
        var suffix: u32 = 1u;
        for (var j = i + 1u; j < NDIM; j++) {
            suffix *= (*shape)[j];
        }
        coords[i] = remaining / suffix;
        remaining = remaining % suffix;
    }
    return coords;
}

// 使用 strides 将 N 维坐标转换为物理偏移量 (支持非连续内存)
fn compute_physical_offset(coords: ptr<function, array<u32, ${ndim}>>, strides: ptr<function, array<i32, ${ndim}>>, base_offset: i32) -> u32 {
    var offset: i32 = base_offset;
    for (var i = 0u; i < NDIM; i++) {
        offset += i32((*coords)[i]) * (*strides)[i];
    }
    return u32(offset);
}
`;
}

/**
 * 生成浮点原子加操作 (CAS loop)
 */
function generateAtomicAddF32(): string {
    return `
// 浮点原子加 (使用 CAS loop)
fn atomicAddF32(idx: u32, val: f32) {
    var old_value = atomicLoad(&output[idx]);
    loop {
        let old_f32 = bitcast<f32>(old_value);
        let new_f32 = old_f32 + val;
        let new_value = bitcast<u32>(new_f32);
        
        let exchange = atomicCompareExchangeWeak(&output[idx], old_value, new_value);
        if (exchange.exchanged) {
            break;
        }
        old_value = exchange.old_value;
    }
}
`;
}

/**
 * 生成浮点原子最大值 (CAS loop)
 */
function generateAtomicMaxF32(): string {
    return `
// 浮点原子最大值 (使用 CAS loop)
fn atomicMaxF32(idx: u32, val: f32) {
    var old_value = atomicLoad(&output[idx]);
    loop {
        let old_f32 = bitcast<f32>(old_value);
        if (val <= old_f32) {
            break;  // val 不大于当前值，无需更新
        }
        let new_value = bitcast<u32>(val);
        
        let exchange = atomicCompareExchangeWeak(&output[idx], old_value, new_value);
        if (exchange.exchanged) {
            break;
        }
        old_value = exchange.old_value;
    }
}
`;
}

/**
 * 生成浮点原子最小值 (CAS loop)
 */
function generateAtomicMinF32(): string {
    return `
// 浮点原子最小值 (使用 CAS loop)
fn atomicMinF32(idx: u32, val: f32) {
    var old_value = atomicLoad(&output[idx]);
    loop {
        let old_f32 = bitcast<f32>(old_value);
        if (val >= old_f32) {
            break;  // val 不小于当前值，无需更新
        }
        let new_value = bitcast<u32>(val);
        
        let exchange = atomicCompareExchangeWeak(&output[idx], old_value, new_value);
        if (exchange.exchanged) {
            break;
        }
        old_value = exchange.old_value;
    }
}
`;
}

/**
 * 生成浮点原子乘法 (CAS loop)
 */
function generateAtomicMulF32(): string {
    return `
// 浮点原子乘法 (使用 CAS loop)
fn atomicMulF32(idx: u32, val: f32) {
    var old_value = atomicLoad(&output[idx]);
    loop {
        let old_f32 = bitcast<f32>(old_value);
        let new_f32 = old_f32 * val;
        let new_value = bitcast<u32>(new_f32);
        
        let exchange = atomicCompareExchangeWeak(&output[idx], old_value, new_value);
        if (exchange.exchanged) {
            break;
        }
        old_value = exchange.old_value;
    }
}
`;
}

/**
 * 生成有符号整数的原子 max/min (比较并交换)
 */
function generateAtomicMaxI32(): string {
    return `
// 有符号整数原子最大值 (CAS loop，因为 atomicMax 是 unsigned)
fn atomicMaxI32(idx: u32, val: i32) {
    var old_value = atomicLoad(&output[idx]);
    loop {
        let old_i32 = bitcast<i32>(old_value);
        if (val <= old_i32) {
            break;
        }
        let new_value = bitcast<u32>(val);
        
        let exchange = atomicCompareExchangeWeak(&output[idx], old_value, new_value);
        if (exchange.exchanged) {
            break;
        }
        old_value = exchange.old_value;
    }
}
`;
}

function generateAtomicMinI32(): string {
    return `
// 有符号整数原子最小值 (CAS loop)
fn atomicMinI32(idx: u32, val: i32) {
    var old_value = atomicLoad(&output[idx]);
    loop {
        let old_i32 = bitcast<i32>(old_value);
        if (val >= old_i32) {
            break;
        }
        let new_value = bitcast<u32>(val);
        
        let exchange = atomicCompareExchangeWeak(&output[idx], old_value, new_value);
        if (exchange.exchanged) {
            break;
        }
        old_value = exchange.old_value;
    }
}
`;
}

/**
 * 生成 scatter (非原子) shader (Strided Implementation)
 *
 * 算法:
 * 1. 先 copy self -> output
 * 2. 对于 index 中的每个位置，将 src 值写入 output 对应位置
 */
export function buildScatterShader(params: ScatterShaderParams): string {
    const {
        selfShape, selfStrides, selfOffset,
        indexShape, indexStrides, indexOffset,
        srcShape, srcStrides, srcOffset,
        outputStrides,
        dim, dtype, indexSize, outputSize
    } = params;
    const ndim = selfShape.length;
    const wType = wgslType(dtype);

    // 使用 i32 存储 strides 以支持负步幅
    const selfShapeStr = `array<u32, ${ndim}>(${selfShape.join(', ')})`;
    const selfStridesStr = `array<i32, ${ndim}>(${selfStrides.join(', ')})`;
    const indexShapeStr = `array<u32, ${ndim}>(${indexShape.join(', ')})`;
    const indexStridesStr = `array<i32, ${ndim}>(${indexStrides.join(', ')})`;
    const srcStridesStr = `array<i32, ${ndim}>(${srcStrides.join(', ')})`;
    const outputStridesStr = `array<i32, ${ndim}>(${outputStrides.join(', ')})`;

    return `
// scatter kernel (Strided Implementation)
// dim = ${dim}, ndim = ${ndim}, indexSize = ${indexSize}, outputSize = ${outputSize}
// Supports non-contiguous input via strides

@group(0) @binding(0) var<storage, read> self_data: array<${wType}>;
@group(0) @binding(1) var<storage, read> index_data: array<i32>;
@group(0) @binding(2) var<storage, read> src_data: array<${wType}>;
@group(0) @binding(3) var<storage, read_write> output: array<${wType}>;

const SELF_SHAPE: array<u32, ${ndim}> = ${selfShapeStr};
const SELF_STRIDES: array<i32, ${ndim}> = ${selfStridesStr};
const SELF_OFFSET: i32 = ${selfOffset};
const INDEX_SHAPE: array<u32, ${ndim}> = ${indexShapeStr};
const INDEX_STRIDES: array<i32, ${ndim}> = ${indexStridesStr};
const INDEX_OFFSET: i32 = ${indexOffset};
const SRC_STRIDES: array<i32, ${ndim}> = ${srcStridesStr};
const SRC_OFFSET: i32 = ${srcOffset};
const OUTPUT_STRIDES: array<i32, ${ndim}> = ${outputStridesStr};
const NDIM: u32 = ${ndim}u;
const DIM: u32 = ${dim}u;
const INDEX_SIZE: u32 = ${indexSize}u;
const OUTPUT_SIZE: u32 = ${outputSize}u;

${generateStridedCoordFunctions(ndim)}

// Phase 1: Copy self -> output (both using strides)
@compute @workgroup_size(256)
fn copy_phase(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= OUTPUT_SIZE) {
        return;
    }
    
    // 计算输出逻辑坐标
    var selfShape = SELF_SHAPE;
    var outCoords = unflatten_logical(idx, &selfShape);
    
    // 计算 self 物理偏移 (使用 strides)
    var selfStrides = SELF_STRIDES;
    let selfPhysical = compute_physical_offset(&outCoords, &selfStrides, SELF_OFFSET);
    
    // 计算 output 物理偏移 (输出总是连续的，但使用通用方法)
    var outputStrides = OUTPUT_STRIDES;
    let outPhysical = compute_physical_offset(&outCoords, &outputStrides, 0);
    
    output[outPhysical] = self_data[selfPhysical];
}

// Phase 2: Scatter src -> output (all using strides)
@compute @workgroup_size(256)
fn scatter_phase(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= INDEX_SIZE) {
        return;
    }
    
    // 计算 index 逻辑坐标
    var indexShape = INDEX_SHAPE;
    let indexCoords = unflatten_logical(idx, &indexShape);
    
    // 从 index 张量读取目标索引 (使用 strided 访问)
    var indexStrides = INDEX_STRIDES;
    var indexCoordsCopy = indexCoords;
    let indexPhysical = compute_physical_offset(&indexCoordsCopy, &indexStrides, INDEX_OFFSET);
    let targetIdx = u32(index_data[indexPhysical]);
    
    // 边界检查
    if (targetIdx >= SELF_SHAPE[DIM]) {
        return;
    }
    
    // 构造输出逻辑坐标
    var outCoords: array<u32, ${ndim}>;
    for (var i = 0u; i < NDIM; i++) {
        if (i == DIM) {
            outCoords[i] = targetIdx;
        } else {
            outCoords[i] = indexCoords[i];
        }
    }
    
    // 计算输出物理偏移
    var outputStrides = OUTPUT_STRIDES;
    let outPhysical = compute_physical_offset(&outCoords, &outputStrides, 0);
    
    // 获取 src 值 (使用 strided 访问)
    var srcStrides = SRC_STRIDES;
    var srcCoords = indexCoords;  // src 和 index 有相同的逻辑坐标范围
    let srcPhysical = compute_physical_offset(&srcCoords, &srcStrides, SRC_OFFSET);
    
    // 写入输出
    output[outPhysical] = src_data[srcPhysical];
}
`;
}

/**
 * 生成 scatter_add shader (使用原子操作, Strided Implementation)
 */
export function buildScatterAddShader(params: ScatterShaderParams): string {
    const {
        selfShape, selfStrides, selfOffset,
        indexShape, indexStrides, indexOffset,
        srcStrides, srcOffset,
        outputStrides,
        dim, dtype, indexSize, outputSize
    } = params;
    const ndim = selfShape.length;
    const wType = wgslType(dtype);
    const isFloat = isFloatType(dtype);
    const isSigned = isSignedIntType(dtype);

    const selfShapeStr = `array<u32, ${ndim}>(${selfShape.join(', ')})`;
    const selfStridesStr = `array<i32, ${ndim}>(${selfStrides.join(', ')})`;
    const indexShapeStr = `array<u32, ${ndim}>(${indexShape.join(', ')})`;
    const indexStridesStr = `array<i32, ${ndim}>(${indexStrides.join(', ')})`;
    const srcStridesStr = `array<i32, ${ndim}>(${srcStrides.join(', ')})`;
    const outputStridesStr = `array<i32, ${ndim}>(${outputStrides.join(', ')})`;

    // 对于原子操作，存储类型统一使用 atomic<u32>
    const storageType = isFloat || isSigned ? 'atomic<u32>' : 'atomic<u32>';

    return `
// scatter_add kernel (Strided, Atomic Implementation)
// dim = ${dim}, ndim = ${ndim}, indexSize = ${indexSize}, outputSize = ${outputSize}
// dtype = ${dtype}, isFloat = ${isFloat}

@group(0) @binding(0) var<storage, read> self_data: array<${wType}>;
@group(0) @binding(1) var<storage, read> index_data: array<i32>;
@group(0) @binding(2) var<storage, read> src_data: array<${wType}>;
@group(0) @binding(3) var<storage, read_write> output: array<${storageType}>;

const SELF_SHAPE: array<u32, ${ndim}> = ${selfShapeStr};
const SELF_STRIDES: array<i32, ${ndim}> = ${selfStridesStr};
const SELF_OFFSET: i32 = ${selfOffset};
const INDEX_SHAPE: array<u32, ${ndim}> = ${indexShapeStr};
const INDEX_STRIDES: array<i32, ${ndim}> = ${indexStridesStr};
const INDEX_OFFSET: i32 = ${indexOffset};
const SRC_STRIDES: array<i32, ${ndim}> = ${srcStridesStr};
const SRC_OFFSET: i32 = ${srcOffset};
const OUTPUT_STRIDES: array<i32, ${ndim}> = ${outputStridesStr};
const NDIM: u32 = ${ndim}u;
const DIM: u32 = ${dim}u;
const INDEX_SIZE: u32 = ${indexSize}u;
const OUTPUT_SIZE: u32 = ${outputSize}u;

${generateStridedCoordFunctions(ndim)}

${isFloat ? generateAtomicAddF32() : ''}

// Phase 1: Copy self -> output (initialize)
@compute @workgroup_size(256)
fn copy_phase(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= OUTPUT_SIZE) {
        return;
    }
    
    var selfShape = SELF_SHAPE;
    let outCoords = unflatten_logical(idx, &selfShape);
    
    var selfStrides = SELF_STRIDES;
    var selfCoordsCopy = outCoords;
    let selfPhysical = compute_physical_offset(&selfCoordsCopy, &selfStrides, SELF_OFFSET);
    
    var outputStrides = OUTPUT_STRIDES;
    var outCoordsCopy = outCoords;
    let outPhysical = compute_physical_offset(&outCoordsCopy, &outputStrides, 0);
    
    // 初始化为 self 的值 (转为 u32 存储)
    ${isFloat
            ? 'atomicStore(&output[outPhysical], bitcast<u32>(self_data[selfPhysical]));'
            : isSigned
                ? 'atomicStore(&output[outPhysical], bitcast<u32>(self_data[selfPhysical]));'
                : 'atomicStore(&output[outPhysical], self_data[selfPhysical]);'
        }
}

// Phase 2: Scatter add
@compute @workgroup_size(256)
fn scatter_add_phase(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= INDEX_SIZE) {
        return;
    }
    
    // 计算 index 逻辑坐标
    var indexShape = INDEX_SHAPE;
    let indexCoords = unflatten_logical(idx, &indexShape);
    
    // 从 index 张量读取目标索引 (使用 strided 访问)
    var indexStrides = INDEX_STRIDES;
    var indexCoordsCopy = indexCoords;
    let indexPhysical = compute_physical_offset(&indexCoordsCopy, &indexStrides, INDEX_OFFSET);
    let targetIdx = u32(index_data[indexPhysical]);
    
    // 边界检查
    if (targetIdx >= SELF_SHAPE[DIM]) {
        return;
    }
    
    // 构造输出逻辑坐标
    var outCoords: array<u32, ${ndim}>;
    for (var i = 0u; i < NDIM; i++) {
        if (i == DIM) {
            outCoords[i] = targetIdx;
        } else {
            outCoords[i] = indexCoords[i];
        }
    }
    
    // 计算输出物理偏移
    var outputStrides = OUTPUT_STRIDES;
    let outPhysical = compute_physical_offset(&outCoords, &outputStrides, 0);
    
    // 获取 src 值 (使用 strided 访问)
    var srcStrides = SRC_STRIDES;
    var srcCoords = indexCoords;
    let srcPhysical = compute_physical_offset(&srcCoords, &srcStrides, SRC_OFFSET);
    let srcVal = src_data[srcPhysical];
    
    // 原子加
    ${isFloat
            ? 'atomicAddF32(outPhysical, srcVal);'
            : isSigned
                ? `{
                var old_value = atomicLoad(&output[outPhysical]);
                loop {
                    let old_val = bitcast<i32>(old_value);
                    let new_val = old_val + srcVal;
                    let exchange = atomicCompareExchangeWeak(&output[outPhysical], old_value, bitcast<u32>(new_val));
                    if (exchange.exchanged) { break; }
                    old_value = exchange.old_value;
                }
            }`
                : 'atomicAdd(&output[outPhysical], srcVal);'
        }
}
`;
}

/**
 * 生成 scatter_reduce shader (Strided Implementation)
 */
export function buildScatterReduceShader(params: ScatterShaderParams): string {
    const {
        config,
        selfShape, selfStrides, selfOffset,
        indexShape, indexStrides, indexOffset,
        srcStrides, srcOffset,
        outputStrides,
        dim, dtype, indexSize, outputSize
    } = params;
    const ndim = selfShape.length;
    const wType = wgslType(dtype);
    const isFloat = isFloatType(dtype);
    const isSigned = isSignedIntType(dtype);
    const reduce = config.reduceMode!;
    const includeSelf = config.includeSelf ?? true;

    const selfShapeStr = `array<u32, ${ndim}>(${selfShape.join(', ')})`;
    const selfStridesStr = `array<i32, ${ndim}>(${selfStrides.join(', ')})`;
    const indexShapeStr = `array<u32, ${ndim}>(${indexShape.join(', ')})`;
    const indexStridesStr = `array<i32, ${ndim}>(${indexStrides.join(', ')})`;
    const srcStridesStr = `array<i32, ${ndim}>(${srcStrides.join(', ')})`;
    const outputStridesStr = `array<i32, ${ndim}>(${outputStrides.join(', ')})`;

    // 需要的原子函数
    let atomicFunctions = '';
    if (isFloat) {
        if (reduce === 'sum' || reduce === 'mean') {
            atomicFunctions = generateAtomicAddF32();
        } else if (reduce === 'prod') {
            atomicFunctions = generateAtomicMulF32();
        } else if (reduce === 'amax') {
            atomicFunctions = generateAtomicMaxF32();
        } else if (reduce === 'amin') {
            atomicFunctions = generateAtomicMinF32();
        }
    } else if (isSigned) {
        if (reduce === 'amax') {
            atomicFunctions = generateAtomicMaxI32();
        } else if (reduce === 'amin') {
            atomicFunctions = generateAtomicMinI32();
        }
    }

    // 生成归约调用
    function getAtomicCall(reduce: ScatterReduceMode): string {
        if (isFloat) {
            switch (reduce) {
                case 'sum':
                case 'mean':
                    return 'atomicAddF32(outPhysical, srcVal);';
                case 'prod':
                    return 'atomicMulF32(outPhysical, srcVal);';
                case 'amax':
                    return 'atomicMaxF32(outPhysical, srcVal);';
                case 'amin':
                    return 'atomicMinF32(outPhysical, srcVal);';
            }
        } else if (isSigned) {
            switch (reduce) {
                case 'sum':
                case 'mean':
                    return `{
                        var old_value = atomicLoad(&output[outPhysical]);
                        loop {
                            let old_val = bitcast<i32>(old_value);
                            let new_val = old_val + srcVal;
                            let exchange = atomicCompareExchangeWeak(&output[outPhysical], old_value, bitcast<u32>(new_val));
                            if (exchange.exchanged) { break; }
                            old_value = exchange.old_value;
                        }
                    }`;
                case 'prod':
                    return `{
                        var old_value = atomicLoad(&output[outPhysical]);
                        loop {
                            let old_val = bitcast<i32>(old_value);
                            let new_val = old_val * srcVal;
                            let exchange = atomicCompareExchangeWeak(&output[outPhysical], old_value, bitcast<u32>(new_val));
                            if (exchange.exchanged) { break; }
                            old_value = exchange.old_value;
                        }
                    }`;
                case 'amax':
                    return 'atomicMaxI32(outPhysical, srcVal);';
                case 'amin':
                    return 'atomicMinI32(outPhysical, srcVal);';
            }
        } else {
            // unsigned int
            switch (reduce) {
                case 'sum':
                case 'mean':
                    return 'atomicAdd(&output[outPhysical], srcVal);';
                case 'prod':
                    return `{
                        var old_value = atomicLoad(&output[outPhysical]);
                        loop {
                            let new_val = old_value * srcVal;
                            let exchange = atomicCompareExchangeWeak(&output[outPhysical], old_value, new_val);
                            if (exchange.exchanged) { break; }
                            old_value = exchange.old_value;
                        }
                    }`;
                case 'amax':
                    return 'atomicMax(&output[outPhysical], srcVal);';
                case 'amin':
                    return 'atomicMin(&output[outPhysical], srcVal);';
            }
        }
        return '';
    }

    // 初始化值
    const identity = getReduceIdentity(reduce, dtype, wType);

    return `
// scatter_reduce kernel (${reduce}, Strided Implementation)
// dim = ${dim}, ndim = ${ndim}, indexSize = ${indexSize}, outputSize = ${outputSize}
// dtype = ${dtype}, reduce = ${reduce}, includeSelf = ${includeSelf}

@group(0) @binding(0) var<storage, read> self_data: array<${wType}>;
@group(0) @binding(1) var<storage, read> index_data: array<i32>;
@group(0) @binding(2) var<storage, read> src_data: array<${wType}>;
@group(0) @binding(3) var<storage, read_write> output: array<atomic<u32>>;

const SELF_SHAPE: array<u32, ${ndim}> = ${selfShapeStr};
const SELF_STRIDES: array<i32, ${ndim}> = ${selfStridesStr};
const SELF_OFFSET: i32 = ${selfOffset};
const INDEX_SHAPE: array<u32, ${ndim}> = ${indexShapeStr};
const INDEX_STRIDES: array<i32, ${ndim}> = ${indexStridesStr};
const INDEX_OFFSET: i32 = ${indexOffset};
const SRC_STRIDES: array<i32, ${ndim}> = ${srcStridesStr};
const SRC_OFFSET: i32 = ${srcOffset};
const OUTPUT_STRIDES: array<i32, ${ndim}> = ${outputStridesStr};
const NDIM: u32 = ${ndim}u;
const DIM: u32 = ${dim}u;
const INDEX_SIZE: u32 = ${indexSize}u;
const OUTPUT_SIZE: u32 = ${outputSize}u;

${generateStridedCoordFunctions(ndim)}

${atomicFunctions}

// Phase 1: Initialize output
@compute @workgroup_size(256)
fn init_phase(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= OUTPUT_SIZE) {
        return;
    }
    
    ${includeSelf
            ? `
    // 初始化为 self 的值 (使用 strided 访问)
    var selfShape = SELF_SHAPE;
    let outCoords = unflatten_logical(idx, &selfShape);
    
    var selfStrides = SELF_STRIDES;
    var selfCoordsCopy = outCoords;
    let selfPhysical = compute_physical_offset(&selfCoordsCopy, &selfStrides, SELF_OFFSET);
    
    var outputStrides = OUTPUT_STRIDES;
    var outCoordsCopy = outCoords;
    let outPhysical = compute_physical_offset(&outCoordsCopy, &outputStrides, 0);
    
    ${isFloat || isSigned
                ? 'atomicStore(&output[outPhysical], bitcast<u32>(self_data[selfPhysical]));'
                : 'atomicStore(&output[outPhysical], self_data[selfPhysical]);'
            }
    `
            : `
    // 初始化为归约单位元
    ${isFloat || isSigned
                ? `atomicStore(&output[idx], bitcast<u32>(${identity}));`
                : `atomicStore(&output[idx], ${identity});`
            }
    `
        }
}

// Phase 2: Scatter reduce
@compute @workgroup_size(256)
fn scatter_reduce_phase(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= INDEX_SIZE) {
        return;
    }
    
    // 计算 index 逻辑坐标
    var indexShape = INDEX_SHAPE;
    let indexCoords = unflatten_logical(idx, &indexShape);
    
    // 从 index 张量读取目标索引 (使用 strided 访问)
    var indexStrides = INDEX_STRIDES;
    var indexCoordsCopy = indexCoords;
    let indexPhysical = compute_physical_offset(&indexCoordsCopy, &indexStrides, INDEX_OFFSET);
    let targetIdx = u32(index_data[indexPhysical]);
    
    // 边界检查
    if (targetIdx >= SELF_SHAPE[DIM]) {
        return;
    }
    
    // 构造输出逻辑坐标
    var outCoords: array<u32, ${ndim}>;
    for (var i = 0u; i < NDIM; i++) {
        if (i == DIM) {
            outCoords[i] = targetIdx;
        } else {
            outCoords[i] = indexCoords[i];
        }
    }
    
    // 计算输出物理偏移
    var outputStrides = OUTPUT_STRIDES;
    let outPhysical = compute_physical_offset(&outCoords, &outputStrides, 0);
    
    // 获取 src 值 (使用 strided 访问)
    var srcStrides = SRC_STRIDES;
    var srcCoords = indexCoords;
    let srcPhysical = compute_physical_offset(&srcCoords, &srcStrides, SRC_OFFSET);
    let srcVal = src_data[srcPhysical];
    
    // 原子归约
    ${getAtomicCall(reduce)}
}
`;
}
