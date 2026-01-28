/**
 * Scatter Shader Builder
 * 
 * Copies data from a contiguous buffer (batch-major) to a strided output buffer.
 * Used to effectively transpose/remap FFT results back to their original dimensional layout.
 */

export interface ScatterConfig {
    workgroupSize: number;
    /** Output dtype elements per pixel (1 for real, 2 for complex) */
    elementsPerPixel: number;
}

export function buildScatterShader(config: ScatterConfig): string {
    const { workgroupSize, elementsPerPixel } = config;

    const useComplex = elementsPerPixel === 2;
    const typeDef = useComplex ? 'vec2<f32>' : 'f32';

    return `
struct Params {
    n: u32,             // Element count along inner dimension (FFT size or OneSided size)
    batch_size: u32,    // Batch size
    stride_outer: u32,  // Stride for outer batch dim
    stride_inner: u32,  // Stride for inner batch dim
    fft_stride: u32,    // Stride for FFT dimension
    batch_inner: u32,   // Size of inner batch
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<${typeDef}>;
@group(0) @binding(1) var<storage, read_write> output: array<${typeDef}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.y;
    let idx = gid.x;
    
    if (batch_idx >= params.batch_size || idx >= params.n) { return; }
    
    // Input is contiguous [Batch, N]
    let in_offset = batch_idx * params.n + idx;
    
    // Decode batch index
    let idx_inner = batch_idx % params.batch_inner;
    let idx_outer = batch_idx / params.batch_inner;

    // Output is strided
    let out_offset = idx_outer * params.stride_outer + idx_inner * params.stride_inner + idx * params.fft_stride;
    
    output[out_offset] = input[in_offset];
}
`;
}
