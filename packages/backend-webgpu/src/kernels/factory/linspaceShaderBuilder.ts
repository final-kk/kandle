
import type { DType } from '@kandle/types';
import { getComputeType } from '../../base/dtype';

/**
 * Build linspace WGSL shader
 * 
 * Generate evenly spaced numbers over a specified interval.
 * Returns numel evenly spaced samples, calculated over the interval [start, end].
 * The endpoint of the interval can optionally be excluded.
 * 
 * logic:
 * if numel <= 1: val = start
 * else: val = start + (idx / (numel - 1)) * (end - start)
 *       or mix(start, end, idx/(numel-1))
 */
export function buildLinspaceShader(outputDtype: DType, workgroupSize: number): string {
    const outputType = getComputeType(outputDtype);

    return `
struct Uniforms {
    numel: u32,
    output_offset: u32,
    start: f32,
    end: f32,
}

@group(0) @binding(0) var<storage, read_write> output: array<${outputType}>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.numel) { return; }
    
    var val: f32;
    if (uniforms.numel <= 1u) {
        val = uniforms.start;
    } else {
        let t = f32(idx) / f32(uniforms.numel - 1u);
        // mix(x, y, a) = x * (1-a) + y * a
        // Accurate linear interpolation
        val = mix(uniforms.start, uniforms.end, t);
    }
    
    output[uniforms.output_offset + idx] = ${outputType}(val);
}
`;
}
