/**
 * Scale Shader Builder for FFT Normalization
 * 
 * Applies normalization scaling to FFT output based on the norm mode:
 * - 'backward': scale = 1 (forward), 1/N (inverse)
 * - 'forward': scale = 1/N (forward), 1 (inverse)
 * - 'ortho': scale = 1/sqrt(N) (both)
 */

export interface ScaleConfig {
    /** Workgroup size */
    workgroupSize: number;
}

/**
 * Build scale shader for FFT normalization
 */
export function buildScaleShader(config: ScaleConfig): string {
    const { workgroupSize } = config;

    return `
// FFT Normalization Scaling
struct Params {
    total_elements: u32,
    scale: f32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    
    if (idx >= params.total_elements) { return; }
    
    data[idx] = data[idx] * params.scale;
}
`;
}
