/**
 * Philox 4x32-10 WGSL Implementation
 * 
 * Counter-based PRNG algorithm used by PyTorch, TensorFlow, and JAX.
 * Reference: "Parallel Random Numbers: As Easy as 1, 2, 3" (SC'11)
 * 
 * PyTorch reference: aten/src/ATen/core/PhiloxRNGEngine.h
 */

/**
 * Philox 4x32-10 核心 WGSL 代码
 * 包含常数定义、辅助函数、核心算法
 */
export const PHILOX_WGSL_CORE = /* wgsl */`
// ========================================
// Philox 4x32-10 Constants
// ========================================

// Philox multiplication constants (carefully selected for statistical quality)
const PHILOX_M0: u32 = 0xD2511F53u;
const PHILOX_M1: u32 = 0xCD9E8D57u;

// Philox key increment constants (derived from golden ratio and sqrt(3)-1)
const PHILOX_W0: u32 = 0x9E3779B9u;  // floor(2^32 / phi)
const PHILOX_W1: u32 = 0xBB67AE85u;  // floor(2^32 * (sqrt(3) - 1) / 2)

// ========================================
// 32×32 → 64-bit Multiplication
// ========================================

/**
 * Compute 32×32 → 64-bit multiplication, returning (hi, lo) as vec2<u32>
 * WGSL doesn't have native 64-bit integers, so we implement this manually.
 * 
 * Algorithm: Schoolbook multiplication in 16-bit chunks
 *   a = a_hi * 2^16 + a_lo
 *   b = b_hi * 2^16 + b_lo
 *   a * b = a_hi * b_hi * 2^32 + (a_hi * b_lo + a_lo * b_hi) * 2^16 + a_lo * b_lo
 */
fn mulhilo32(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;
    
    // Partial products
    let lo_lo = a_lo * b_lo;           // up to 32 bits
    let hi_lo = a_hi * b_lo;           // up to 32 bits
    let lo_hi = a_lo * b_hi;           // up to 32 bits
    let hi_hi = a_hi * b_hi;           // up to 32 bits
    
    // Cross-term accumulation (carefully handle overflow)
    let cross = (lo_lo >> 16u) + (hi_lo & 0xFFFFu) + lo_hi;
    let hi = (hi_lo >> 16u) + (cross >> 16u) + hi_hi;
    let lo = (cross << 16u) | (lo_lo & 0xFFFFu);
    
    return vec2<u32>(hi, lo);
}

// ========================================
// Philox Single Round
// ========================================

/**
 * Philox 4x32 single round transformation
 * 
 * Performs:
 * 1. Two 32×32→64 multiplications
 * 2. XOR operations with counter elements and key
 * 3. Swapping/permutation of elements
 */
fn philox4x32_round(counter: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    // Multiply counter[0] and counter[2] with Philox constants
    let product0 = mulhilo32(PHILOX_M0, counter.x);  // (hi0, lo0)
    let product1 = mulhilo32(PHILOX_M1, counter.z);  // (hi1, lo1)
    
    // Apply Feistel-like mixing with key
    return vec4<u32>(
        product1.x ^ counter.y ^ key.x,  // hi1 ^ c1 ^ k0
        product1.y,                       // lo1
        product0.x ^ counter.w ^ key.y,  // hi0 ^ c3 ^ k1
        product0.y                        // lo0
    );
}

// ========================================
// Philox 4x32-10 Full Algorithm
// ========================================

/**
 * Philox 4x32-10: 10-round Philox algorithm
 * 
 * Given a 128-bit counter and 64-bit key, produces 128 bits (4 × u32) of random output.
 * The 10-round variant is the standard choice, balancing quality and performance.
 */
fn philox4x32_10(counter: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    var ctr = counter;
    var k = key;
    
    // 10 rounds with key increment between each
    ctr = philox4x32_round(ctr, k); k = vec2<u32>(k.x + PHILOX_W0, k.y + PHILOX_W1);
    ctr = philox4x32_round(ctr, k); k = vec2<u32>(k.x + PHILOX_W0, k.y + PHILOX_W1);
    ctr = philox4x32_round(ctr, k); k = vec2<u32>(k.x + PHILOX_W0, k.y + PHILOX_W1);
    ctr = philox4x32_round(ctr, k); k = vec2<u32>(k.x + PHILOX_W0, k.y + PHILOX_W1);
    ctr = philox4x32_round(ctr, k); k = vec2<u32>(k.x + PHILOX_W0, k.y + PHILOX_W1);
    ctr = philox4x32_round(ctr, k); k = vec2<u32>(k.x + PHILOX_W0, k.y + PHILOX_W1);
    ctr = philox4x32_round(ctr, k); k = vec2<u32>(k.x + PHILOX_W0, k.y + PHILOX_W1);
    ctr = philox4x32_round(ctr, k); k = vec2<u32>(k.x + PHILOX_W0, k.y + PHILOX_W1);
    ctr = philox4x32_round(ctr, k); k = vec2<u32>(k.x + PHILOX_W0, k.y + PHILOX_W1);
    ctr = philox4x32_round(ctr, k);
    // Note: No key increment after the last round
    
    return ctr;
}

// ========================================
// Distribution Conversion Functions
// ========================================

/**
 * Convert u32 to uniform float in [0, 1)
 * 
 * PyTorch approach: (x >> 8) * (1.0 / 2^24)
 * This preserves 24 bits of precision (matches float32 mantissa)
 * Range: [0, 1 - 2^-24) ≈ [0, 0.99999994)
 */
fn u32_to_uniform(x: u32) -> f32 {
    // 5.9604644775390625e-8 = 1.0 / 16777216.0 = 1.0 / 2^24
    return f32(x >> 8u) * 5.9604644775390625e-8;
}

/**
 * Constants for Box-Muller transform
 */
const TWO_PI: f32 = 6.283185307179586;

/**
 * Box-Muller transform: convert two uniform [0,1) to one normal N(0,1)
 * 
 * Given u1, u2 ~ Uniform(0,1):
 *   z0 = sqrt(-2 * ln(u1)) * cos(2π * u2)
 *   z1 = sqrt(-2 * ln(u1)) * sin(2π * u2)
 * 
 * We only use z0 (cos variant) for simplicity.
 */
fn box_muller(u1: f32, u2: f32) -> f32 {
    // Guard against log(0) - clamp u1 to a small positive value
    let safe_u1 = max(u1, 1e-10);
    
    let r = sqrt(-2.0 * log(safe_u1));
    let theta = TWO_PI * u2;
    
    return r * cos(theta);
}
`;

/**
 * Uniform buffer 声明
 */
export const RANDOM_UNIFORMS_WGSL = /* wgsl */`
struct RandomUniforms {
    // Basic info (16 bytes, vec4 aligned)
    numel: u32,
    output_offset: u32,
    _pad0: u32,
    _pad1: u32,
    
    // Philox Key (16 bytes)
    key0: u32,
    key1: u32,
    base_offset: u32,
    _pad2: u32,
    
    // randint params (16 bytes)
    low: i32,
    high: i32,
    _pad3: u32,
    _pad4: u32,
};

@group(0) @binding(0) var<uniform> uniforms: RandomUniforms;
`;
