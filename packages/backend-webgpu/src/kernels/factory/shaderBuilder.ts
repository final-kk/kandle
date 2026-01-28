/**
 * Factory Shader Builder
 * 
 * 生成工厂操作的 WGSL shader
 * 
 * @module kernels/factory/shaderBuilder
 */

import type { DType } from '@kandle/types';
import { getComputeType } from '../../base/dtype';
import { Logger } from '@kandle/utils';

const logger = new Logger('Factory-ShaderBuilder');

/**
 * 构建 Eye (单位矩阵) shader
 * 
 * 逻辑: if (row == col) output[idx] = 1; else output[idx] = 0;
 */
export function buildEyeShader(
    n: number,
    m: number,
    outputDtype: DType,
    workgroupSize: number
): string {
    const outputType = getComputeType(outputDtype);

    logger.debug(`Building Eye shader: n=${n}, m=${m}, dtype=${outputDtype}`);

    return `
// ============================================================================
// Eye Matrix: Identity matrix (n x m)
// ============================================================================

struct Uniforms {
    numel: u32,     // 总元素数 (n * m)
    n: u32,         // 行数
    m: u32,         // 列数
    outputOffset: u32,
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
    
    // 计算 (row, col) 坐标
    let row = idx / uniforms.m;
    let col = idx % uniforms.m;
    
    // 对角线为 1，其余为 0
    var value: ${outputType};
    if (row == col) {
        value = ${outputType}(1);
    } else {
        value = ${outputType}(0);
    }
    
    output[uniforms.outputOffset + idx] = value;
}
`;
}
