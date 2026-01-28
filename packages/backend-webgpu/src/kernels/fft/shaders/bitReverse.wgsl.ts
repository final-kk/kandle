/**
 * Bit Reversal Shader Builder
 * 
 * Performs bit-reversal permutation required for Cooley-Tukey FFT.
 * Input indices are reordered based on their bit-reversed values.
 * 
 * Supports strided input for arbitrary-dimension FFT:
 * - fft_stride: stride along the FFT dimension
 * - outer_stride: stride for outer (batch) dimensions
 * - Output is always contiguous
 */


export interface BitReverseConfig {
    /** Workgroup size */
    workgroupSize: number;
    /** Whether input is real (needs conversion to complex) */
    isRealInput: boolean;
    /** Input dtype */
    inputDtype: 'complex64' | 'complex128' | 'float32' | 'float64';
}

/**
 * Build bit-reversal shader
 * 
 * This shader reads from input buffer (strided), applies bit-reversal permutation,
 * and writes to output buffer (contiguous). If input is real, it converts to complex.
 * 
 * Input layout: input[batch_idx * outer_stride + fft_idx * fft_stride]
 * Output layout: output[batch_idx * n + reversed_fft_idx] (always contiguous)
 */
export function buildBitReverseShader(config: BitReverseConfig): string {
    const { workgroupSize, isRealInput } = config;

    // For complex types, we use vec2<f32> (complex64) or vec2<f32> (complex128 degraded)
    const complexType = 'vec2<f32>';
    const realType = 'f32';

    if (isRealInput) {
        return `
// Bit-Reversal Permutation with Real-to-Complex Conversion (Strided Input)
struct Params {
    n: u32,           // FFT size
    log2n: u32,       // log2(n)
    batch_size: u32,  // Total number of FFT batches
    stride_outer: u32,// Stride for outer batch dims
    stride_inner: u32,// Stride for inner batch dims
    fft_stride: u32,  // Stride along FFT dimension
    batch_inner: u32, // Size of inner batch
    base_offset: u32, // Input tensor base offset (in logical elements)
}

@group(0) @binding(0) var<storage, read> input: array<${realType}>;
@group(0) @binding(1) var<storage, read_write> output: array<${complexType}>;
@group(0) @binding(2) var<uniform> params: Params;

fn bitReverse(x: u32, bits: u32) -> u32 {
    var result: u32 = 0u;
    var val = x;
    for (var i: u32 = 0u; i < bits; i++) {
        result = (result << 1u) | (val & 1u);
        val >>= 1u;
    }
    return result;
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.y;
    let fft_idx = gid.x;
    
    if (batch_idx >= params.batch_size || fft_idx >= params.n) { return; }
    
    let reversed_idx = bitReverse(fft_idx, params.log2n);
    
    // Decode batch index into outer and inner components
    // batch_idx corresponds to flattened [batch_outer, batch_inner]
    let idx_inner = batch_idx % params.batch_inner;
    let idx_outer = batch_idx / params.batch_inner;

    // Input offset: base_offset + outer_idx * stride_outer + inner_idx * stride_inner + fft_idx * fft_stride
    let in_offset = params.base_offset + idx_outer * params.stride_outer + idx_inner * params.stride_inner + fft_idx * params.fft_stride;
    
    // Output offset: batch * n + reversed_idx (CONTIGUOUS)
    let out_offset = batch_idx * params.n + reversed_idx;
    
    // Convert real to complex (imag = 0)
    output[out_offset] = ${complexType}(input[in_offset], 0.0);
}
`;
    } else {
        return `
// Bit-Reversal Permutation for Complex Input (Strided Input)
struct Params {
    n: u32,           // FFT size
    log2n: u32,       // log2(n)
    batch_size: u32,  // Total number of FFT batches
    stride_outer: u32,// Stride for outer batch dims
    stride_inner: u32,// Stride for inner batch dims
    fft_stride: u32,  // Stride along FFT dimension
    batch_inner: u32, // Size of inner batch
    base_offset: u32, // Input tensor base offset (in logical elements)
}

@group(0) @binding(0) var<storage, read> input: array<${complexType}>;
@group(0) @binding(1) var<storage, read_write> output: array<${complexType}>;
@group(0) @binding(2) var<uniform> params: Params;

fn bitReverse(x: u32, bits: u32) -> u32 {
    var result: u32 = 0u;
    var val = x;
    for (var i: u32 = 0u; i < bits; i++) {
        result = (result << 1u) | (val & 1u);
        val >>= 1u;
    }
    return result;
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.y;
    let fft_idx = gid.x;
    
    if (batch_idx >= params.batch_size || fft_idx >= params.n) { return; }
    
    let reversed_idx = bitReverse(fft_idx, params.log2n);
    
    // Decode batch index into outer and inner components
    let idx_inner = batch_idx % params.batch_inner;
    let idx_outer = batch_idx / params.batch_inner;

    // Input offset: base_offset + outer_idx * stride_outer + inner_idx * stride_inner + fft_idx * fft_stride
    let in_offset = params.base_offset + idx_outer * params.stride_outer + idx_inner * params.stride_inner + fft_idx * params.fft_stride;
    
    // Output offset: batch * n + reversed_idx (CONTIGUOUS)
    let out_offset = batch_idx * params.n + reversed_idx;
    
    output[out_offset] = input[in_offset];
}
`;
    }
}
