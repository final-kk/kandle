/**
 * Arange Shader Builder
 * 
 * 生成 arange 操作的 WGSL shader
 * 
 * @module kernels/factory/arangeShaderBuilder
 */

import type { DType } from '@kandle/types';
import { getComputeType } from '../../base/dtype';
import { Logger } from '@kandle/utils';

const logger = new Logger('Arange-ShaderBuilder');

/**
 * 判断 dtype 是否为整数类型
 * 
 * 用于决定 uniform buffer 中 start/step 的存储格式
 */
export function isIntegerDtype(dtype: DType): boolean {
    return dtype === 'int8' || dtype === 'int16' || dtype === 'int32' || dtype === 'int64' ||
        dtype === 'uint8' || dtype === 'uint16' || dtype === 'uint32' || dtype === 'uint64';
}

/**
 * Build arange WGSL shader
 * 
 * 生成等差数列: output[i] = start + i * step
 * 
 * @param outputDtype - Output dtype (determines storage and compute types)
 * @param workgroupSize - Workgroup size
 * @returns WGSL shader code
 */
export function buildArangeShader(
    outputDtype: DType,
    workgroupSize: number
): string {
    const isInteger = isIntegerDtype(outputDtype);
    const outputType = getComputeType(outputDtype);

    logger.debug(`Building Arange shader: dtype=${outputDtype}, isInteger=${isInteger}`);

    // 对于整数类型，uniform 中的 start/step 存储为 i32
    // 对于浮点类型，uniform 中的 start/step 存储为 f32
    const uniformStartType = isInteger ? 'i32' : 'f32';

    return `
// ============================================================================
// Arange: Generate arithmetic sequence [start, start+step, start+2*step, ...]
// ============================================================================

struct Uniforms {
    numel: u32,           // Total number of elements
    output_offset: u32,   // Offset in output buffer
    start: ${uniformStartType},  // Start value
    step: ${uniformStartType},   // Step value
}

@group(0) @binding(0) var<storage, read_write> output: array<${outputType}>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(${workgroupSize})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    
    if (idx >= uniforms.numel) {
        return;
    }
    
    // Compute value: start + idx * step
    ${isInteger ? `
    let value = uniforms.start + ${uniformStartType}(idx) * uniforms.step;
    output[uniforms.output_offset + idx] = ${outputType}(value);
    ` : `
    let value = uniforms.start + f32(idx) * uniforms.step;
    output[uniforms.output_offset + idx] = ${outputType}(value);
    `}
}
`;
}
