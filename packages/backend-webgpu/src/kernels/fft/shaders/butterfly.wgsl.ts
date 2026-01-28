/**
 * Butterfly Operation Shader Builder
 * 
 * Implements the butterfly operation for Cooley-Tukey radix-2 FFT.
 * Each stage processes pairs of elements with increasing distance.
 */

export interface ButterflyConfig {
    /** Workgroup size */
    workgroupSize: number;
}

/**
 * Build butterfly shader for FFT stages
 * 
 * The butterfly operation for each stage:
 * - Pairs have distance 2^stage
 * - Each pair (top, bot) is updated as:
 *   top' = top + twiddle * bot
 *   bot' = top - twiddle * bot
 * 
 * Twiddle factors are computed inline using Euler's formula.
 */
export function buildButterflyShader(config: ButterflyConfig): string {
    const { workgroupSize } = config;

    return `
// FFT Butterfly Operation - Single Stage
struct Params {
    n: u32,
    stage: u32,
    direction: f32,  // -1.0 for forward, 1.0 for inverse
    batch_size: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> twiddle_factors: array<vec2<f32>>;

// Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.y;
    let butterfly_idx = gid.x;
    
    let num_butterflies = params.n >> 1u;
    
    if (batch_idx >= params.batch_size || butterfly_idx >= num_butterflies) { 
        return; 
    }
    
    // Calculate indices for this butterfly
    let block_size = 1u << (params.stage + 1u);
    let half_block = block_size >> 1u;
    
    let block_idx = butterfly_idx / half_block;
    let pair_idx = butterfly_idx % half_block;
    
    let top_idx = block_idx * block_size + pair_idx;
    let bot_idx = top_idx + half_block;
    
    // Add batch offset
    let batch_offset = batch_idx * params.n;
    let top_global = batch_offset + top_idx;
    let bot_global = batch_offset + bot_idx;
    
    // Fetch twiddle factor from pre-computed buffer
    // k = pair_idx * (N / block_size)
    // We need integer division. params.n and block_size are floats? No u32.
    let k = pair_idx * (params.n / block_size);
    
    let w = twiddle_factors[k];
    
    // Adjust for direction
    // Buffer stores Forward factors (exp(-i*angle))
    // Forward (dir=-1): Use w (mult imag by 1 = -dir)
    // Inverse (dir=1): Use conj(w) (mult imag by -1 = -dir)
    let twiddle = vec2<f32>(w.x, w.y * -params.direction);
    
    // Load values
    let top = data[top_global];
    let bot = data[bot_global];
    
    // Butterfly operation
    let twiddle_bot = cmul(bot, twiddle);
    data[top_global] = top + twiddle_bot;
    data[bot_global] = top - twiddle_bot;
}
`;
}
