import { DType } from "@kandle/types";
import { getGlobalDTypeResolver } from "../base/DTypeResolver";
import { WGSL_CONSTANTS } from "../base/dtype";

/**
 * 辅助接口：加载函数的生成结果
 */
export interface LoadSnippetResult {
    funcName: string;    // 函数名
    code: string;        // 完整代码
    returnType: string;  // 该函数返回值的 WGSL 类型 (e.g. 'u32', 'f32')
}

/**
 * 1. 获取 WebGPU Storage Buffer 的声明类型
 * 使用 DTypeResolver 获取物理存储类型
 */
export function getStorageType(dtype: DType): string {
    const resolver = getGlobalDTypeResolver();
    const descriptor = resolver.getDescriptor(dtype);
    return `array<${descriptor.wgslStorageType}>`;
}

/**
 * 2. 生成读取数据的辅助函数 (Snippet)
 * 职责：从 Storage 读取(物理) -> 解包/转换 -> 返回(逻辑)
 * 
 * 重要：此函数假设数据已经在上传时被正确转换为物理格式
 * - uint8/int8: 已 pack 为 u32 (4 个 byte -> 1 个 u32)
 * - float64: 已降级为 f32
 * - int64/uint64: 已降级为 i32/u32
 * - int16/uint16: 已扩展为 i32/u32
 */
export function generateLoadSnippet(name: string, dtype: DType): LoadSnippetResult {
    const funcName = `load_${name}`;

    // 物理存储类型 (e.g. 'u32') - 使用 DTypeResolver
    const resolver = getGlobalDTypeResolver();
    const descriptor = resolver.getDescriptor(dtype);
    const storageType = descriptor.wgslStorageType;

    // 逻辑返回类型 (默认与物理类型一致，但以下情况例外)
    let returnType = storageType; // 默认为物理类型

    // 根据逻辑类型修正返回类型
    switch (dtype) {
        case 'bool': returnType = 'bool'; break; // 物理 u32 -> 逻辑 bool
        case 'int8': returnType = 'i32'; break;  // 物理 u32(packed) -> 逻辑 i32
        case 'uint8': returnType = 'u32'; break;  // 物理 u32(packed) -> 逻辑 u32
        case 'int16': returnType = 'i32'; break;  // 物理 i32 -> 逻辑 i32
        case 'uint16': returnType = 'u32'; break;  // 物理 u32 -> 逻辑 u32
        case 'float32': returnType = 'f32'; break;
        case 'int32': returnType = 'i32'; break;
        case 'uint32': returnType = 'u32'; break;
        case 'float64': returnType = 'f32'; break; // 物理 f32 (降级后)
        case 'int64': returnType = 'i32'; break; // 物理 i32 (降级后)
        case 'uint64': returnType = 'u32'; break; // 物理 u32 (降级后)
        case 'float16': returnType = storageType; break; // 从 DTypeResolver 获取 (f16 或 f32)
        case 'complex64': returnType = 'vec2<f32>'; break;  // 存储为 vec2<f32>
        case 'complex128': returnType = 'vec2<f32>'; break; // 降级后也是 vec2<f32>
    }

    let body = '';

    // === 情况 A: 简单标量 (float32, int32, uint32, 以及降级后的 float64, int64, uint64, float16) ===
    if (['float32', 'int32', 'uint32', 'float64', 'int64', 'uint64', 'int16', 'uint16', 'float16'].includes(dtype)) {
        // 直接读取，无需位操作
        // 注意：对于降级类型，数据已经在 JS 端转换过了，shader 里直接按物理类型读取
        body = `return ${name}[idx];`;
    }
    // === 情况 B: 布尔值 (bool) ===
    else if (dtype === 'bool') {
        // u32 (0/1) -> bool
        body = `return ${name}[idx] != 0u;`;
    }
    // === 情况 C: Int8 (Expanded in u32) ===
    else if (dtype === 'int8') {
        body = `return ${name}[idx];`; // Reads i32
    }
    // === 情况 D: Uint8 (Expanded in u32) ===
    else if (dtype === 'uint8') {
        body = `return ${name}[idx];`; // Reads u32
    }
    // === 情况 E: Complex (vec2<f32>) ===
    else if (dtype === 'complex64' || dtype === 'complex128') {
        // Complex 存储为 array<vec2<f32>>，直接读取
        body = `return ${name}[idx];`;
    }
    else {
        throw new Error(`[ShaderSnippets] Load generation not implemented for ${dtype}`);
    }

    const code = `
    fn ${funcName}(idx: u32) -> ${returnType} {
        ${body}
    }
    `;

    return { funcName, code, returnType };
}

/**
 * 3. 生成存储数据的辅助函数 (Snippet)
 * 职责：将计算结果写入 Storage Buffer
 * 
 * 注意：对于 packed 类型 (int8, uint8)，需要特殊处理
 */
export function generateStoreSnippet(name: string, dtype: DType): { funcName: string, code: string } {
    const funcName = `store_${name}`;

    // 大多数类型直接存储
    if (['float32', 'int32', 'uint32', 'float64', 'int64', 'uint64', 'int16', 'uint16', 'bool'].includes(dtype)) {
        // 对于降级类型，存储时也是按物理类型存储
        // bool 存储为 u32
        const storeExpr = dtype === 'bool'
            ? `${name}[idx] = select(0u, 1u, value);`
            : `${name}[idx] = value;`;

        const paramType = dtype === 'bool' ? 'bool' : getGlobalDTypeResolver().getDescriptor(dtype).wgslStorageType;

        return {
            funcName,
            code: `
            fn ${funcName}(idx: u32, value: ${paramType}) {
                ${storeExpr}
            }
            `
        };
    }

    // Packed int8 - 需要原子操作或单独处理
    // 目前简化：假设输出不是 int8/uint8，如果需要支持，需要更复杂的逻辑
    // Expanded int8/uint8 - 直接存储 (as i32/u32)
    if (dtype === 'int8' || dtype === 'uint8') {
        const paramType = dtype === 'int8' ? 'i32' : 'u32';
        return {
            funcName,
            code: `
             fn ${funcName}(idx: u32, value: ${paramType}) {
                 ${name}[idx] = value;
             }
             `
        };
    }

    throw new Error(`[ShaderSnippets] Store generation not implemented for ${dtype}`);
}

/**
 * 4. 生成坐标映射函数 (Offset Calculation)
 * (保持原有逻辑，确保和 Builder 对接)
 */
export function generateOffsetFunction(name: string, rank: number): { funcName: string, code: string } {
    const funcName = `get_offset_${name}`;

    if (rank === 0) {
        return { funcName, code: `fn ${funcName}(idx: u32) -> u32 { return 0u; }` };
    }

    // 反向遍历 (Row-Major)
    let calculations = `
        var offset = 0u;
        var current_idx = idx;
    `;

    for (let i = rank - 1; i >= 0; i--) {
        calculations += `
        {
            let dim_size = uniforms.shape[${i}];
            let stride = uniforms.strides_${name}[${i}];
            let coord = current_idx % dim_size;
            current_idx = current_idx / dim_size;
            offset = offset + coord * stride;
        }
        `;
    }

    const code = `
    fn ${funcName}(idx: u32) -> u32 {
        ${calculations}
        return offset;
    }
    `;

    return { funcName, code };
}

/**
 * 5. Generate reduction initializer
 * Returns the WGSL code to initialize the accumulator variable
 */
export function generateReductionInitializer(opName: string, computeType: string): string {
    switch (opName) {
        case 'sum':
        case 'mean':
        case 'nansum':
        case 'nanmean':
            return `var acc: ${computeType} = ${computeType}(0.0);`;
        case 'prod':
            return `var acc: ${computeType} = ${computeType}(1.0);`;
        case 'max':
            if (computeType === 'f32' || computeType === 'f16') {
                return `var acc: ${computeType} = ${computeType}(${WGSL_CONSTANTS.NEG_FLT_MAX});`;
            } else {
                return `var acc: ${computeType} = ${computeType}(${WGSL_CONSTANTS.INT_MIN});`;
            }
        case 'min':
            if (computeType === 'f32' || computeType === 'f16') {
                return `var acc: ${computeType} = ${computeType}(${WGSL_CONSTANTS.FLT_MAX});`;
            } else {
                return `var acc: ${computeType} = ${computeType}(${WGSL_CONSTANTS.INT_MAX});`;
            }
        default:
            throw new Error(`[ShaderSnippets] Unknown reduction op for initializer: ${opName}`);
    }
}

/**
 * 6. Generate reduction accumulate operation
 * Returns the WGSL code to combine the accumulated value with a new value
 * @param opName - Reduction operation name
 * @param handleNaN - Whether to handle NaN values (for nansum, nanmean)
 */
export function generateReductionAccumulate(opName: string, handleNaN = false): string {
    const nanCheck = handleNaN ? 'if (!isnan(val)) { ' : '';
    const nanClose = handleNaN ? ' }' : '';

    switch (opName) {
        case 'sum':
        case 'mean':
            return `${nanCheck}acc = acc + val;${nanClose}`;
        case 'nansum':
        case 'nanmean':
            return generateReductionAccumulate(opName.replace('nan', '') as any, true);
        case 'prod':
            return `acc = acc * val;`;
        case 'max':
            return `acc = max(acc, val);`;
        case 'min':
            return `acc = min(acc, val);`;
        default:
            throw new Error(`[ShaderSnippets] Unknown reduction op for accumulate: ${opName}`);
    }
}

/**
 * 7. Generate reduction finalizer
 * Returns the WGSL code to finalize the reduction result (e.g., divide by count for mean)
 * NOTE: Uses runtime uniforms.reduction_numel instead of compile-time constant
 */
export function generateReductionFinalizer(opName: string): string {
    switch (opName) {
        case 'mean':
        case 'nanmean':
            // Mean requires dividing by the number of elements
            // Use runtime uniform value to ensure correctness
            return `let result = acc / f32(uniforms.reduction_numel);`;
        case 'sum':
        case 'nansum':
        case 'prod':
        case 'max':
        case 'min':
            // No post-processing needed
            return `let result = acc;`;
        default:
            throw new Error(`[ShaderSnippets] Unknown reduction op for finalizer: ${opName}`);
    }
}

/**
 * 8. Generate reduction offset function (for inner reduction loop)
 * Similar to generateOffsetFunction but for reduction dimensions
 */
export function generateReductionOffsetFunction(name: string, reductionRank: number): { funcName: string, code: string } {
    const funcName = `get_reduction_offset_${name}`;

    if (reductionRank === 0) {
        return { funcName, code: `fn ${funcName}(idx: u32) -> u32 { return 0u; }` };
    }

    // Calculates offset within reduction dimensions
    let calculations = `
        var offset = 0u;
        var current_idx = idx;
    `;

    for (let i = reductionRank - 1; i >= 0; i--) {
        calculations += `
        {
            let dim_size = uniforms.reduction_shape[${i}];
            let stride = uniforms.reduction_strides_${name}[${i}];
            let coord = current_idx % dim_size;
            current_idx = current_idx / dim_size;
            offset = offset + coord * stride;
        }
        `;
    }

    const code = `
    fn ${funcName}(idx: u32) -> u32 {
        ${calculations}
        return offset;
    }
    `;

    return { funcName, code };
}
