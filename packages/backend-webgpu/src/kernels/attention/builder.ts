/**
 * FlashAttention WGSL Shader Builder
 * 
 * 生成 FlashAttention 的 WGSL compute shader
 * 
 * 核心算法:
 * 1. Tiling: 按 block 迭代 K/V，每个 workgroup 处理一个 Q block
 * 2. Online Softmax: 维护 running max (m) 和 sum (l) 完成增量 softmax
 * 3. Causal mask: 通过条件分支跳过 masked blocks
 */

import type { DType } from '@kandle/types';
import { getComputeType } from '../../base/dtype';
import type { FlashAttentionTileConfig } from './types';

// ============================================================================
// Shader Configuration
// ============================================================================

export interface FlashAttentionShaderConfig {
    dtype: DType;
    headDim: number;
    blockSizeQ: number;
    blockSizeKV: number;
    isCausal: boolean;
    /** GQA ratio: numHeadsQ / numHeadsKV */
    gqaRatio: number;
}

// ============================================================================
// WGSL Type Helpers
// ============================================================================

function getWgslType(dtype: DType): string {
    return getComputeType(dtype);
}

function getWgslStorageType(dtype: DType): string {
    // For most types, storage type is the same as compute type
    // Float16 would use f16 if enabled, but we default to f32
    return getComputeType(dtype);
}

// ============================================================================
// Shader Builder
// ============================================================================

/**
 * 构建 FlashAttention WGSL Shader
 */
export function buildFlashAttentionShader(
    config: FlashAttentionShaderConfig,
    tileConfig: FlashAttentionTileConfig
): string {
    const { dtype, headDim, blockSizeQ, blockSizeKV, isCausal, gqaRatio } = config;
    const { workgroupSizeX, workgroupSizeY } = tileConfig;

    const wgslType = getWgslType(dtype);
    const wgslStorageType = getWgslStorageType(dtype);

    // 计算 shared memory 大小
    // Q_block: [blockSizeQ, headDim]
    // K_block: [blockSizeKV, headDim]
    // V_block: [blockSizeKV, headDim]
    // S_block: [blockSizeQ, blockSizeKV] - attention scores
    // O_acc: [blockSizeQ, headDim] - output accumulator
    // m: [blockSizeQ] - running max
    // l: [blockSizeQ] - running sum of exp

    return `
// ============================================================================
// FlashAttention Kernel
// Block size Q: ${blockSizeQ}, Block size KV: ${blockSizeKV}
// Head dim: ${headDim}, Causal: ${isCausal}, GQA ratio: ${gqaRatio}
// ============================================================================

struct Uniforms {
    // vec4<u32> [0]: dimensions
    batchSize: u32,
    numHeadsQ: u32,
    numHeadsKV: u32,
    headDim: u32,
    
    // vec4<u32> [1]: sequence lengths
    seqLenQ: u32,
    seqLenKV: u32,
    blockSizeQ: u32,
    blockSizeKV: u32,
    
    // vec4 [2]: scale and offsets
    scale: f32,
    offsetQ: u32,
    offsetK: u32,
    offsetV: u32,
    
    // vec4<u32> [3]: output offset and flags
    offsetO: u32,
    isCausal: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> Q: array<${wgslStorageType}>;
@group(0) @binding(2) var<storage, read> K: array<${wgslStorageType}>;
@group(0) @binding(3) var<storage, read> V: array<${wgslStorageType}>;
@group(0) @binding(4) var<storage, read_write> O: array<${wgslStorageType}>;

// Workgroup shared memory
var<workgroup> Q_shared: array<${wgslType}, ${blockSizeQ * headDim}>;
var<workgroup> K_shared: array<${wgslType}, ${blockSizeKV * headDim}>;
var<workgroup> V_shared: array<${wgslType}, ${blockSizeKV * headDim}>;
var<workgroup> S_shared: array<${wgslType}, ${blockSizeQ * blockSizeKV}>;

// Per-row statistics for online softmax
var<workgroup> m_shared: array<${wgslType}, ${blockSizeQ}>;  // running max
var<workgroup> l_shared: array<${wgslType}, ${blockSizeQ}>;  // running sum of exp

// Output accumulator
var<workgroup> O_shared: array<${wgslType}, ${blockSizeQ * headDim}>;

const BLOCK_SIZE_Q: u32 = ${blockSizeQ}u;
const BLOCK_SIZE_KV: u32 = ${blockSizeKV}u;
const HEAD_DIM: u32 = ${headDim}u;
const NEG_INF: ${wgslType} = ${wgslType}(-1e20);

@compute @workgroup_size(${workgroupSizeX}, ${workgroupSizeY})
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
) {
    // Workgroup mapping:
    // workgroup_id.x = Q block index
    // workgroup_id.y = head index (Q heads)
    // workgroup_id.z = batch index
    
    let q_block_idx = workgroup_id.x;
    let head_q = workgroup_id.y;
    let batch = workgroup_id.z;
    
    // GQA: map Q head to KV head
    let head_kv = head_q / ${gqaRatio}u;
    
    let local_row = local_id.x;  // within Q block
    let local_col = local_id.y;  // within head dim
    
    // Global Q row index
    let q_row_start = q_block_idx * BLOCK_SIZE_Q;
    
    // Base offsets for this batch/head
    let q_base = uniforms.offsetQ + batch * uniforms.numHeadsQ * uniforms.seqLenQ * HEAD_DIM 
                 + head_q * uniforms.seqLenQ * HEAD_DIM;
    let k_base = uniforms.offsetK + batch * uniforms.numHeadsKV * uniforms.seqLenKV * HEAD_DIM 
                 + head_kv * uniforms.seqLenKV * HEAD_DIM;
    let v_base = uniforms.offsetV + batch * uniforms.numHeadsKV * uniforms.seqLenKV * HEAD_DIM 
                 + head_kv * uniforms.seqLenKV * HEAD_DIM;
    let o_base = uniforms.offsetO + batch * uniforms.numHeadsQ * uniforms.seqLenQ * HEAD_DIM 
                 + head_q * uniforms.seqLenQ * HEAD_DIM;
    
    // ========================================================================
    // Step 1: Load Q block into shared memory
    // ========================================================================
    for (var i = local_idx; i < BLOCK_SIZE_Q * HEAD_DIM; i += ${workgroupSizeX * workgroupSizeY}u) {
        let row = i / HEAD_DIM;
        let col = i % HEAD_DIM;
        let global_row = q_row_start + row;
        
        if (global_row < uniforms.seqLenQ) {
            Q_shared[i] = ${wgslType}(Q[q_base + global_row * HEAD_DIM + col]);
        } else {
            Q_shared[i] = ${wgslType}(0.0);
        }
    }
    
    // Initialize output accumulator, m, l
    for (var i = local_idx; i < BLOCK_SIZE_Q * HEAD_DIM; i += ${workgroupSizeX * workgroupSizeY}u) {
        O_shared[i] = ${wgslType}(0.0);
    }
    for (var i = local_idx; i < BLOCK_SIZE_Q; i += ${workgroupSizeX * workgroupSizeY}u) {
        m_shared[i] = NEG_INF;
        l_shared[i] = ${wgslType}(0.0);
    }
    
    workgroupBarrier();
    
    // ========================================================================
    // Step 2: Iterate over KV blocks
    // ========================================================================
    let num_kv_blocks = (uniforms.seqLenKV + BLOCK_SIZE_KV - 1u) / BLOCK_SIZE_KV;
    
    for (var kv_block_idx = 0u; kv_block_idx < num_kv_blocks; kv_block_idx++) {
        let kv_start = kv_block_idx * BLOCK_SIZE_KV;
        
        // Causal mask: skip blocks that are entirely masked
        ${isCausal ? `
        // For causal attention, Q row i can only attend to K cols <= i
        // If the entire KV block is after the Q block, skip it
        let q_row_max = q_row_start + BLOCK_SIZE_Q - 1u;
        if (kv_start > q_row_max) {
            continue;
        }
        ` : ''}
        
        // Load K block
        for (var i = local_idx; i < BLOCK_SIZE_KV * HEAD_DIM; i += ${workgroupSizeX * workgroupSizeY}u) {
            let row = i / HEAD_DIM;
            let col = i % HEAD_DIM;
            let global_row = kv_start + row;
            
            if (global_row < uniforms.seqLenKV) {
                K_shared[i] = ${wgslType}(K[k_base + global_row * HEAD_DIM + col]);
            } else {
                K_shared[i] = ${wgslType}(0.0);
            }
        }
        
        // Load V block
        for (var i = local_idx; i < BLOCK_SIZE_KV * HEAD_DIM; i += ${workgroupSizeX * workgroupSizeY}u) {
            let row = i / HEAD_DIM;
            let col = i % HEAD_DIM;
            let global_row = kv_start + row;
            
            if (global_row < uniforms.seqLenKV) {
                V_shared[i] = ${wgslType}(V[v_base + global_row * HEAD_DIM + col]);
            } else {
                V_shared[i] = ${wgslType}(0.0);
            }
        }
        
        workgroupBarrier();
        
        // ====================================================================
        // Step 3: Compute S = Q @ K^T (with scaling)
        // ====================================================================
        for (var i = local_idx; i < BLOCK_SIZE_Q * BLOCK_SIZE_KV; i += ${workgroupSizeX * workgroupSizeY}u) {
            let q_row = i / BLOCK_SIZE_KV;
            let k_col = i % BLOCK_SIZE_KV;
            let global_q_row = q_row_start + q_row;
            let global_k_col = kv_start + k_col;
            
            var score = ${wgslType}(0.0);
            
            // Check bounds
            if (global_q_row < uniforms.seqLenQ && global_k_col < uniforms.seqLenKV) {
                // Apply causal mask
                ${isCausal ? `
                if (global_k_col > global_q_row) {
                    score = NEG_INF;
                } else {
                ` : ''}
                    // Dot product: Q[q_row] @ K[k_col]
                    for (var d = 0u; d < HEAD_DIM; d++) {
                        score += Q_shared[q_row * HEAD_DIM + d] * K_shared[k_col * HEAD_DIM + d];
                    }
                    score *= ${wgslType}(uniforms.scale);
                ${isCausal ? '}' : ''}
            } else {
                score = NEG_INF;
            }
            
            S_shared[i] = score;
        }
        
        workgroupBarrier();
        
        // ====================================================================
        // Step 4: Online Softmax update
        // ====================================================================
        for (var q_row = local_idx; q_row < BLOCK_SIZE_Q; q_row += ${workgroupSizeX * workgroupSizeY}u) {
            let global_q_row = q_row_start + q_row;
            if (global_q_row >= uniforms.seqLenQ) {
                continue;
            }
            
            // Find max in this row of S
            var m_new = m_shared[q_row];
            for (var k = 0u; k < BLOCK_SIZE_KV; k++) {
                let s = S_shared[q_row * BLOCK_SIZE_KV + k];
                m_new = max(m_new, s);
            }
            
            // Compute l_new = l_old * exp(m_old - m_new) + sum(exp(S - m_new))
            let m_old = m_shared[q_row];
            let l_old = l_shared[q_row];
            let scale_factor = exp(m_old - m_new);
            
            var l_new = l_old * scale_factor;
            for (var k = 0u; k < BLOCK_SIZE_KV; k++) {
                let s = S_shared[q_row * BLOCK_SIZE_KV + k];
                l_new += exp(s - m_new);
            }
            
            // Update O accumulator: O = O * scale_factor + sum(P * V)
            // where P[i] = exp(S[i] - m_new) / l_new (normalized later)
            for (var d = 0u; d < HEAD_DIM; d++) {
                var o_val = O_shared[q_row * HEAD_DIM + d] * scale_factor;
                
                for (var k = 0u; k < BLOCK_SIZE_KV; k++) {
                    let s = S_shared[q_row * BLOCK_SIZE_KV + k];
                    let p = exp(s - m_new);
                    o_val += p * V_shared[k * HEAD_DIM + d];
                }
                
                O_shared[q_row * HEAD_DIM + d] = o_val;
            }
            
            // Store updated m, l
            m_shared[q_row] = m_new;
            l_shared[q_row] = l_new;
        }
        
        workgroupBarrier();
    }
    
    // ========================================================================
    // Step 5: Normalize and write output
    // ========================================================================
    for (var i = local_idx; i < BLOCK_SIZE_Q * HEAD_DIM; i += ${workgroupSizeX * workgroupSizeY}u) {
        let q_row = i / HEAD_DIM;
        let d = i % HEAD_DIM;
        let global_q_row = q_row_start + q_row;
        
        if (global_q_row < uniforms.seqLenQ) {
            let l = l_shared[q_row];
            let o_val = O_shared[i] / l;
            O[o_base + global_q_row * HEAD_DIM + d] = ${wgslStorageType}(o_val);
        }
    }
}
`;
}
