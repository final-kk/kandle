/**
 * Scan Shader Builder (v5)
 * 
 * Generates WGSL shaders for scan (prefix sum) operations
 * 
 * Algorithm: Blelloch Work-Efficient Parallel Scan
 * Reference: GPU Gems 3 Chapter 39 - Parallel Prefix Sum (Scan) with CUDA
 * 
 * Supports:
 * - Single-pass scan for small dimensions (scanDimSize <= workgroupSize)
 * - Multi-pass scan for large dimensions (up-sweep, down-sweep, block add)
 * - cumsum, cumprod (single output)
 * - cummax, cummin (dual output: values + indices)
 */

import { ITensorIterator } from '@kandle/types';
import { Logger } from '@kandle/utils';
import { getStorageType, generateLoadSnippet } from '../../shader/ShaderSnippets';
import { getComputeType, generateCastSnippet } from '../../base/dtype';
import { getGlobalDTypeResolver } from '../../base/DTypeResolver';
import { SCAN_OPS } from './ops';
import { ScanDimParams } from './types';

const logger = new Logger('Scan-ShaderBuilder');

// ============================================================================
// Bank Conflict Avoidance
// ============================================================================

/**
 * WebGPU shared memory has 32 banks (typically)
 * To avoid bank conflicts in tree algorithms, we add padding
 */
const NUM_BANKS = 32;
const LOG_NUM_BANKS = 5;

/**
 * Generate conflict-free offset macro for WGSL
 * This adds n >> 5 = n / 32 padding elements
 */
function conflictFreeOffset(n: string): string {
    return `((${n}) >> ${LOG_NUM_BANKS}u)`;
}

// ============================================================================
// Single-Pass Scan Shader (for small scanDimSize)
// ============================================================================

/**
 * Build single-pass scan shader using Blelloch algorithm
 * 
 * Each workgroup handles one "slice" through the tensor perpendicular to scan dim
 * The workgroup cooperatively scans along the scan dimension
 * 
 * @param iter - TensorIterator with scan configuration
 * @param dispatchKey - Operation name: 'cumsum' | 'cumprod' | 'cummax' | 'cummin'
 * @param workgroupSize - Number of threads per workgroup
 * @param params - Scan dimension parameters
 */
export function buildSinglePassScanShader(
    iter: ITensorIterator,
    dispatchKey: string,
    workgroupSize: number,
    params: ScanDimParams
): string {
    const opConfig = SCAN_OPS[dispatchKey];
    if (!opConfig) {
        throw new Error(`Unknown scan operation: ${dispatchKey}`);
    }

    const input = iter.input(0);
    const output = iter.output();
    const resolver = getGlobalDTypeResolver();
    const computeType = getComputeType(iter.computeDtype);
    const storageTypeInput = getStorageType(input.dtype);
    const storageTypeOutput = getStorageType(output.dtype);
    const elemTypeInput = resolver.getDescriptor(input.dtype).wgslStorageType;
    const elemTypeOutput = resolver.getDescriptor(output.dtype).wgslStorageType;

    const loaderInput = generateLoadSnippet('input', input.dtype);

    const { scanDimSize, outerSize, innerSize } = params;

    // Shared memory needs to hold scanDimSize elements + padding for bank conflict avoidance
    // Round up to next power of 2 for tree algorithm
    // Minimum of 2 to ensure half_n >= 1 (at least one thread participates)
    const sharedMemSize = Math.max(2, Math.pow(2, Math.ceil(Math.log2(scanDimSize))));
    const paddedSharedSize = sharedMemSize + Math.ceil(sharedMemSize / NUM_BANKS);

    const needsF16 = input.dtype === 'float16';
    const enableF16 = needsF16 && resolver.supportsNativeF16 ? 'enable f16;\n' : '';

    // Build up-sweep (reduce) phase: log2(n) iterations
    // d = 0: stride = 1, pairs at (1,2), (3,4), ...
    // d = 1: stride = 2, pairs at (3,4), (7,8), ...
    // ...
    const upSweepCode = buildUpSweepCode(sharedMemSize, opConfig.operator, computeType);

    // Build down-sweep phase: log2(n) iterations in reverse
    // First, set last element to identity
    // Then propagate down
    const downSweepCode = buildDownSweepCode(sharedMemSize, opConfig.operator, opConfig.identity, computeType);

    return `
${enableF16}
// Single-Pass Scan Shader: ${dispatchKey}
// Input shape: [${iter.inputShape.join(', ')}]
// Scan dim: ${params.scanDim}, size: ${scanDimSize}
// Outer: ${outerSize}, Inner: ${innerSize}

struct Uniforms {
    scan_dim_size: u32,       // Size of dimension being scanned
    outer_size: u32,          // Product of dims before scan dim
    inner_size: u32,          // Product of dims after scan dim
    scan_dim_stride: u32,     // Stride of scan dimension
    inner_stride: u32,        // Stride of innermost dimensions (typically 1)
    input_offset: u32,
    output_offset: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: ${storageTypeInput};
@group(0) @binding(2) var<storage, read_write> output: ${storageTypeOutput};

// Shared memory with bank conflict avoidance padding
var<workgroup> shared_data: array<${computeType}, ${paddedSharedSize}>;

${loaderInput.code}

// Conflict-free index calculation
fn cf_idx(n: u32) -> u32 {
    return n + (n >> ${LOG_NUM_BANKS}u);
}

@compute @workgroup_size(${workgroupSize})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    
    // Each workgroup handles one "line" through the tensor
    // wid.x = outer_idx * inner_size + inner_idx
    let slice_idx = wid.x;
    let outer_idx = slice_idx / uniforms.inner_size;
    let inner_idx = slice_idx % uniforms.inner_size;
    
    if (outer_idx >= uniforms.outer_size) {
        return;
    }
    
    // Base offset for this slice in input/output
    // For a tensor indexed as [outer...][scan][inner...]
    // offset = outer_idx * (scan_dim_size * inner_size) + inner_idx
    //        = outer_idx * scan_dim_stride * scan_dim_size + inner_idx * inner_stride
    // But we also need to handle the actual stride pattern
    let base_offset = outer_idx * uniforms.scan_dim_stride * uniforms.scan_dim_size 
                    + inner_idx * uniforms.inner_stride;
    
    // === Step 1: Load input data into shared memory ===
    // Blelloch algorithm: n/2 threads process n elements
    // Each participating thread loads 2 elements: index i and index i + n/2
    let n = ${sharedMemSize}u;
    let half_n = n / 2u;
    
    // Only first n/2 threads participate in load/store
    // They load elements at positions tid and tid + n/2
    if (tid < half_n) {
        let ai = tid;
        let bi = tid + half_n;
        
        // Load first half
        if (ai < uniforms.scan_dim_size) {
            let input_idx = uniforms.input_offset + base_offset + ai * uniforms.scan_dim_stride;
            let raw_val = ${loaderInput.funcName}(input_idx);
            shared_data[cf_idx(ai)] = ${generateCastSnippet('raw_val', elemTypeInput, computeType)};
        } else {
            shared_data[cf_idx(ai)] = ${opConfig.identity(computeType)};
        }
        
        // Load second half
        if (bi < uniforms.scan_dim_size) {
            let input_idx = uniforms.input_offset + base_offset + bi * uniforms.scan_dim_stride;
            let raw_val = ${loaderInput.funcName}(input_idx);
            shared_data[cf_idx(bi)] = ${generateCastSnippet('raw_val', elemTypeInput, computeType)};
        } else {
            shared_data[cf_idx(bi)] = ${opConfig.identity(computeType)};
        }
    }
    
    // === Step 2: Up-sweep (Reduce) Phase ===
${upSweepCode}
    
    // === Step 3: Set root to identity (for exclusive scan) ===
    // We'll convert to inclusive at the end
    workgroupBarrier();
    if (tid == 0u) {
        // Store the total sum before clearing for exclusive scan
        // This is needed to convert exclusive to inclusive
        shared_data[cf_idx(n - 1u)] = ${opConfig.identity(computeType)};
    }
    
    // === Step 4: Down-sweep Phase ===
${downSweepCode}
    
    workgroupBarrier();
    
    // === Step 5: Write output (convert exclusive to inclusive) ===
    // Inclusive scan: output[i] = exclusive[i] + input[i]
    // Only first n/2 threads participate in write-back
    // This is the standard way to get inclusive from exclusive
    
    if (tid < half_n) {
        let ai = tid;
        let bi = tid + half_n;
        
        if (ai < uniforms.scan_dim_size) {
            let input_idx = uniforms.input_offset + base_offset + ai * uniforms.scan_dim_stride;
            let output_idx = uniforms.output_offset + base_offset + ai * uniforms.scan_dim_stride;
            let raw_val = ${loaderInput.funcName}(input_idx);
            let input_val = ${generateCastSnippet('raw_val', elemTypeInput, computeType)};
            let exclusive_val = shared_data[cf_idx(ai)];
            let inclusive_val = ${opConfig.operator('exclusive_val', 'input_val', computeType)};
            output[output_idx] = ${computeType === elemTypeOutput ? 'inclusive_val' : generateCastSnippet('inclusive_val', computeType as any, elemTypeOutput)};
        }
        
        if (bi < uniforms.scan_dim_size) {
            let input_idx = uniforms.input_offset + base_offset + bi * uniforms.scan_dim_stride;
            let output_idx = uniforms.output_offset + base_offset + bi * uniforms.scan_dim_stride;
            let raw_val = ${loaderInput.funcName}(input_idx);
            let input_val = ${generateCastSnippet('raw_val', elemTypeInput, computeType)};
            let exclusive_val = shared_data[cf_idx(bi)];
            let inclusive_val = ${opConfig.operator('exclusive_val', 'input_val', computeType)};
            output[output_idx] = ${computeType === elemTypeOutput ? 'inclusive_val' : generateCastSnippet('inclusive_val', computeType as any, elemTypeOutput)};
        }
    }
}
`;
}

/**
 * Generate up-sweep (reduce) phase WGSL code
 * This is the first half of Blelloch's algorithm
 */
function buildUpSweepCode(n: number, operator: (a: string, b: string, t: string) => string, computeType: string): string {
    const logN = Math.log2(n);
    let code = '';

    for (let d = 0; d < logN; d++) {
        const stride = 1 << d;  // 2^d
        const numActive = n >> (d + 1);  // n / 2^(d+1)

        code += `
    // Up-sweep step ${d}: stride = ${stride}
    workgroupBarrier();
    {
        let d_stride = ${stride}u;
        let k = tid;
        if (k < ${numActive}u) {
            let ai = d_stride * (2u * k + 1u) - 1u;
            let bi = d_stride * (2u * k + 2u) - 1u;
            shared_data[cf_idx(bi)] = ${operator('shared_data[cf_idx(ai)]', 'shared_data[cf_idx(bi)]', computeType)};
        }
    }`;
    }

    return code;
}

/**
 * Generate down-sweep phase WGSL code
 * This is the second half of Blelloch's algorithm
 */
function buildDownSweepCode(n: number, operator: (a: string, b: string, t: string) => string, identity: (t: string) => string, computeType: string): string {
    const logN = Math.log2(n);
    let code = '';

    for (let d = logN - 1; d >= 0; d--) {
        const stride = 1 << d;
        const numActive = n >> (d + 1);

        code += `
    // Down-sweep step ${logN - 1 - d}: stride = ${stride}
    workgroupBarrier();
    {
        let d_stride = ${stride}u;
        let k = tid;
        if (k < ${numActive}u) {
            let ai = d_stride * (2u * k + 1u) - 1u;
            let bi = d_stride * (2u * k + 2u) - 1u;
            let t = shared_data[cf_idx(ai)];
            shared_data[cf_idx(ai)] = shared_data[cf_idx(bi)];
            shared_data[cf_idx(bi)] = ${operator('t', 'shared_data[cf_idx(bi)]', computeType)};
        }
    }`;
    }

    return code;
}

// ============================================================================
// Multi-Pass Scan Shaders (for large scanDimSize)
// ============================================================================

/**
 * Build Stage 1 shader: Scan blocks and output block sums
 * Each workgroup scans WORKGROUP_SIZE * 2 elements
 */
export function buildMultiPassScanStage1(
    iter: ITensorIterator,
    dispatchKey: string,
    workgroupSize: number,
    params: ScanDimParams,
    elementsPerBlock: number
): string {
    const opConfig = SCAN_OPS[dispatchKey];
    if (!opConfig) {
        throw new Error(`Unknown scan operation: ${dispatchKey}`);
    }

    const input = iter.input(0);
    const output = iter.output();
    const resolver = getGlobalDTypeResolver();
    const computeType = getComputeType(iter.computeDtype);
    const storageTypeInput = getStorageType(input.dtype);
    const storageTypeOutput = getStorageType(output.dtype);
    const elemTypeInput = resolver.getDescriptor(input.dtype).wgslStorageType;
    const elemTypeOutput = resolver.getDescriptor(output.dtype).wgslStorageType;

    const loaderInput = generateLoadSnippet('input', input.dtype);

    const { scanDimSize, outerSize, innerSize } = params;

    // Shared memory for block scan
    const paddedSharedSize = elementsPerBlock + Math.ceil(elementsPerBlock / NUM_BANKS);

    const needsF16 = input.dtype === 'float16';
    const enableF16 = needsF16 && resolver.supportsNativeF16 ? 'enable f16;\n' : '';

    const upSweepCode = buildUpSweepCode(elementsPerBlock, opConfig.operator, computeType);
    const downSweepCode = buildDownSweepCode(elementsPerBlock, opConfig.operator, opConfig.identity, computeType);

    const numBlocks = Math.ceil(scanDimSize / elementsPerBlock);

    return `
${enableF16}
// Multi-Pass Scan Stage 1: ${dispatchKey}
// Block scan with block sum output
// Elements per block: ${elementsPerBlock}, Num blocks: ${numBlocks}

struct Uniforms {
    scan_dim_size: u32,
    num_blocks: u32,
    elements_per_block: u32,
    outer_size: u32,
    inner_size: u32,
    scan_dim_stride: u32,
    inner_stride: u32,
    input_offset: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: ${storageTypeInput};
@group(0) @binding(2) var<storage, read_write> scanned_blocks: ${storageTypeOutput};
@group(0) @binding(3) var<storage, read_write> block_sums: array<${computeType}>;

var<workgroup> shared_data: array<${computeType}, ${paddedSharedSize}>;

${loaderInput.code}

fn cf_idx(n: u32) -> u32 {
    return n + (n >> ${LOG_NUM_BANKS}u);
}

@compute @workgroup_size(${workgroupSize})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    
    // wid.x encodes: slice_idx * num_blocks + block_idx
    let total_blocks_per_slice = uniforms.num_blocks;
    let global_block_idx = wid.x;
    let slice_idx = global_block_idx / total_blocks_per_slice;
    let block_idx = global_block_idx % total_blocks_per_slice;
    
    let outer_idx = slice_idx / uniforms.inner_size;
    let inner_idx = slice_idx % uniforms.inner_size;
    
    if (outer_idx >= uniforms.outer_size) {
        return;
    }
    
    let base_offset = outer_idx * uniforms.scan_dim_stride * uniforms.scan_dim_size 
                    + inner_idx * uniforms.inner_stride;
    let block_start = block_idx * uniforms.elements_per_block;
    
    let n = ${elementsPerBlock}u;
    let half_n = n / 2u;
    
    // Only first n/2 threads participate in load/store (Blelloch algorithm)
    if (tid < half_n) {
        let ai = tid;
        let bi = tid + half_n;
        
        // Load with bounds check
        let global_ai = block_start + ai;
        let global_bi = block_start + bi;
        
        if (global_ai < uniforms.scan_dim_size) {
            let input_idx = uniforms.input_offset + base_offset + global_ai * uniforms.scan_dim_stride;
            let raw_val = ${loaderInput.funcName}(input_idx);
            shared_data[cf_idx(ai)] = ${generateCastSnippet('raw_val', elemTypeInput, computeType)};
        } else {
            shared_data[cf_idx(ai)] = ${opConfig.identity(computeType)};
        }
        
        if (global_bi < uniforms.scan_dim_size) {
            let input_idx = uniforms.input_offset + base_offset + global_bi * uniforms.scan_dim_stride;
            let raw_val = ${loaderInput.funcName}(input_idx);
            shared_data[cf_idx(bi)] = ${generateCastSnippet('raw_val', elemTypeInput, computeType)};
        } else {
            shared_data[cf_idx(bi)] = ${opConfig.identity(computeType)};
        }
    }
    
    // Up-sweep
${upSweepCode}
    
    workgroupBarrier();
    
    // Store block sum before clearing last element
    if (tid == 0u) {
        block_sums[global_block_idx] = shared_data[cf_idx(n - 1u)];
        shared_data[cf_idx(n - 1u)] = ${opConfig.identity(computeType)};
    }
    
    // Down-sweep
${downSweepCode}
    
    workgroupBarrier();
    
    // Write scanned result (still exclusive at this point)
    // Only first n/2 threads participate in write-back
    // We'll add block prefix and convert to inclusive in Stage 3
    if (tid < half_n) {
        let ai = tid;
        let bi = tid + half_n;
        let global_ai = block_start + ai;
        let global_bi = block_start + bi;
        
        if (global_ai < uniforms.scan_dim_size) {
            let output_idx = base_offset + global_ai * uniforms.scan_dim_stride;
            let exclusive_val = shared_data[cf_idx(ai)];
            scanned_blocks[output_idx] = ${computeType === elemTypeOutput ? 'exclusive_val' : generateCastSnippet('exclusive_val', computeType as any, elemTypeOutput)};
        }
        
        if (global_bi < uniforms.scan_dim_size) {
            let output_idx = base_offset + global_bi * uniforms.scan_dim_stride;
            let exclusive_val = shared_data[cf_idx(bi)];
            scanned_blocks[output_idx] = ${computeType === elemTypeOutput ? 'exclusive_val' : generateCastSnippet('exclusive_val', computeType as any, elemTypeOutput)};
        }
    }
}
`;
}

/**
 * Build Stage 2 shader: Scan the block sums
 * This is just a regular scan on the block_sums array
 */
export function buildMultiPassScanStage2(
    dispatchKey: string,
    computeType: string,
    workgroupSize: number,
    numBlocks: number,
    numSlices: number
): string {
    const opConfig = SCAN_OPS[dispatchKey];
    if (!opConfig) {
        throw new Error(`Unknown scan operation: ${dispatchKey}`);
    }

    // We need to scan numBlocks elements per slice
    const sharedMemSize = Math.pow(2, Math.ceil(Math.log2(numBlocks)));
    const paddedSharedSize = sharedMemSize + Math.ceil(sharedMemSize / NUM_BANKS);

    const upSweepCode = buildUpSweepCode(sharedMemSize, opConfig.operator, computeType);
    const downSweepCode = buildDownSweepCode(sharedMemSize, opConfig.operator, opConfig.identity, computeType);

    return `
// Multi-Pass Scan Stage 2: Scan block sums
// Num blocks per slice: ${numBlocks}, Num slices: ${numSlices}

struct Uniforms {
    num_blocks: u32,
    num_slices: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> block_sums: array<${computeType}>;
@group(0) @binding(2) var<storage, read_write> block_prefixes: array<${computeType}>;

var<workgroup> shared_data: array<${computeType}, ${paddedSharedSize}>;

fn cf_idx(n: u32) -> u32 {
    return n + (n >> ${LOG_NUM_BANKS}u);
}

@compute @workgroup_size(${workgroupSize})
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let slice_idx = wid.x;
    
    if (slice_idx >= uniforms.num_slices) {
        return;
    }
    
    let base = slice_idx * uniforms.num_blocks;
    let n = ${sharedMemSize}u;
    let half_n = n / 2u;
    
    // Only first n/2 threads participate in load/store (Blelloch algorithm)
    if (tid < half_n) {
        let ai = tid;
        let bi = tid + half_n;
        
        // Load block sums
        if (ai < uniforms.num_blocks) {
            shared_data[cf_idx(ai)] = block_sums[base + ai];
        } else {
            shared_data[cf_idx(ai)] = ${opConfig.identity(computeType)};
        }
        
        if (bi < uniforms.num_blocks) {
            shared_data[cf_idx(bi)] = block_sums[base + bi];
        } else {
            shared_data[cf_idx(bi)] = ${opConfig.identity(computeType)};
        }
    }
    
    // Up-sweep
${upSweepCode}
    
    workgroupBarrier();
    if (tid == 0u) {
        shared_data[cf_idx(n - 1u)] = ${opConfig.identity(computeType)};
    }
    
    // Down-sweep
${downSweepCode}
    
    workgroupBarrier();
    
    // Write block prefixes (exclusive scan of block sums)
    // Only first n/2 threads participate in write-back
    if (tid < half_n) {
        let ai = tid;
        let bi = tid + half_n;
        
        if (ai < uniforms.num_blocks) {
            block_prefixes[base + ai] = shared_data[cf_idx(ai)];
        }
        if (bi < uniforms.num_blocks) {
            block_prefixes[base + bi] = shared_data[cf_idx(bi)];
        }
    }
}
`;
}

/**
 * Build Stage 3 shader: Add block prefixes and convert to inclusive scan
 */
export function buildMultiPassScanStage3(
    iter: ITensorIterator,
    dispatchKey: string,
    workgroupSize: number,
    params: ScanDimParams,
    elementsPerBlock: number
): string {
    const opConfig = SCAN_OPS[dispatchKey];
    if (!opConfig) {
        throw new Error(`Unknown scan operation: ${dispatchKey}`);
    }

    const input = iter.input(0);
    const output = iter.output();
    const resolver = getGlobalDTypeResolver();
    const computeType = getComputeType(iter.computeDtype);
    const storageTypeInput = getStorageType(input.dtype);
    const storageTypeOutput = getStorageType(output.dtype);
    const elemTypeInput = resolver.getDescriptor(input.dtype).wgslStorageType;
    const elemTypeOutput = resolver.getDescriptor(output.dtype).wgslStorageType;

    const loaderInput = generateLoadSnippet('input', input.dtype);

    const { scanDimSize, outerSize, innerSize } = params;
    const numBlocks = Math.ceil(scanDimSize / elementsPerBlock);

    const needsF16 = input.dtype === 'float16';
    const enableF16 = needsF16 && resolver.supportsNativeF16 ? 'enable f16;\n' : '';

    return `
${enableF16}
// Multi-Pass Scan Stage 3: Add block prefixes and convert to inclusive
// Elements per block: ${elementsPerBlock}

struct Uniforms {
    scan_dim_size: u32,
    num_blocks: u32,
    elements_per_block: u32,
    outer_size: u32,
    inner_size: u32,
    scan_dim_stride: u32,
    inner_stride: u32,
    input_offset: u32,
    output_offset: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: ${storageTypeInput};
@group(0) @binding(2) var<storage, read> scanned_blocks: ${storageTypeOutput};
@group(0) @binding(3) var<storage, read> block_prefixes: array<${computeType}>;
@group(0) @binding(4) var<storage, read_write> output: ${storageTypeOutput};

${loaderInput.code}

@compute @workgroup_size(${workgroupSize})
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    // Each thread handles elements_per_block / workgroupSize elements
    let global_idx = gid.x;
    
    // Decode indices
    let slice_idx = global_idx / uniforms.scan_dim_size;
    let elem_idx = global_idx % uniforms.scan_dim_size;
    
    let outer_idx = slice_idx / uniforms.inner_size;
    let inner_idx = slice_idx % uniforms.inner_size;
    
    if (outer_idx >= uniforms.outer_size) {
        return;
    }
    
    let base_offset = outer_idx * uniforms.scan_dim_stride * uniforms.scan_dim_size 
                    + inner_idx * uniforms.inner_stride;
    
    // Which block does this element belong to?
    let block_idx = elem_idx / uniforms.elements_per_block;
    let global_block_idx = slice_idx * uniforms.num_blocks + block_idx;
    
    // Get the scanned value (exclusive) and block prefix
    let scan_idx = base_offset + elem_idx * uniforms.scan_dim_stride;
    let exclusive_val = ${computeType}(scanned_blocks[scan_idx]);
    let block_prefix = block_prefixes[global_block_idx];
    
    // Add block prefix to get global exclusive prefix
    let global_exclusive = ${opConfig.operator('block_prefix', 'exclusive_val', computeType)};
    
    // Add original input value to convert to inclusive
    let input_idx = uniforms.input_offset + scan_idx;
    let raw_input = ${loaderInput.funcName}(input_idx);
    let input_val = ${generateCastSnippet('raw_input', elemTypeInput, computeType)};
    let inclusive_val = ${opConfig.operator('global_exclusive', 'input_val', computeType)};
    
    // Write output
    let output_idx = uniforms.output_offset + scan_idx;
    output[output_idx] = ${computeType === elemTypeOutput ? 'inclusive_val' : generateCastSnippet('inclusive_val', computeType as any, elemTypeOutput)};
}
`;
}

// ============================================================================
// Cummax/Cummin Shader (with indices)
// ============================================================================

/**
 * Build scan shader for cummax/cummin that outputs both values and indices
 * Uses a sequential approach for correctness (parallel argmax/argmin is complex)
 */
export function buildCumExtremumShader(
    iter: ITensorIterator,
    dispatchKey: string,
    workgroupSize: number,
    params: ScanDimParams
): string {
    const opConfig = SCAN_OPS[dispatchKey];
    if (!opConfig || !opConfig.compare) {
        throw new Error(`Unknown cummax/cummin operation or missing compare: ${dispatchKey}`);
    }

    const input = iter.input(0);
    const valuesOutput = iter.output(0);
    const indicesOutput = iter.output(1);

    const resolver = getGlobalDTypeResolver();
    const computeType = getComputeType(iter.computeDtype);
    const storageTypeInput = getStorageType(input.dtype);
    const storageTypeValues = getStorageType(valuesOutput.dtype);
    const storageTypeIndices = getStorageType(indicesOutput.dtype);
    const elemTypeInput = resolver.getDescriptor(input.dtype).wgslStorageType;
    const elemTypeValues = resolver.getDescriptor(valuesOutput.dtype).wgslStorageType;

    const loaderInput = generateLoadSnippet('input', input.dtype);

    const { scanDimSize, outerSize, innerSize } = params;

    const needsF16 = input.dtype === 'float16';
    const enableF16 = needsF16 && resolver.supportsNativeF16 ? 'enable f16;\n' : '';

    // For cummax/cummin, we use a simpler sequential approach per slice
    // Each workgroup handles multiple slices
    return `
${enableF16}
// Cumulative Extremum Shader: ${dispatchKey}
// Input shape: [${iter.inputShape.join(', ')}]
// Scan dim: ${params.scanDim}, size: ${scanDimSize}

struct Uniforms {
    scan_dim_size: u32,
    outer_size: u32,
    inner_size: u32,
    scan_dim_stride: u32,
    inner_stride: u32,
    input_offset: u32,
    output_values_offset: u32,
    output_indices_offset: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> input: ${storageTypeInput};
@group(0) @binding(2) var<storage, read_write> output_values: ${storageTypeValues};
@group(0) @binding(3) var<storage, read_write> output_indices: ${storageTypeIndices};

${loaderInput.code}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slice_idx = gid.x;
    
    let outer_idx = slice_idx / uniforms.inner_size;
    let inner_idx = slice_idx % uniforms.inner_size;
    
    if (outer_idx >= uniforms.outer_size) {
        return;
    }
    
    let base_offset = outer_idx * uniforms.scan_dim_stride * uniforms.scan_dim_size 
                    + inner_idx * uniforms.inner_stride;
    
    // Initialize with first element
    var current_val = ${opConfig.identity(computeType)};
    var current_idx: u32 = 0u;
    
    // Sequential scan along dimension
    for (var i: u32 = 0u; i < uniforms.scan_dim_size; i = i + 1u) {
        let input_idx = uniforms.input_offset + base_offset + i * uniforms.scan_dim_stride;
        let raw_val = ${loaderInput.funcName}(input_idx);
        let val = ${generateCastSnippet('raw_val', elemTypeInput, computeType)};
        
        // Update if new value satisfies compare condition
        if (${opConfig.compare('val', 'current_val')}) {
            current_val = val;
            current_idx = i;
        }
        
        // Write output
        let output_idx = base_offset + i * uniforms.scan_dim_stride;
        output_values[uniforms.output_values_offset + output_idx] = ${computeType === elemTypeValues ? 'current_val' : generateCastSnippet('current_val', computeType as any, elemTypeValues)};
        // Note: int64 is degraded to i32 on WebGPU, so we cast to i32
        output_indices[uniforms.output_indices_offset + output_idx] = i32(current_idx);
    }
}
`;
}
