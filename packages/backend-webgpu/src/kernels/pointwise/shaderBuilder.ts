/**
 * v5 Pointwise Shader Builder
 * 
 * Generates WGSL shader code for pointwise operations
 * Supports both contiguous (fast path) and strided (general path) access
 * 
 * IMPORTANT: Uses correct WGSL memory alignment for uniform buffers
 * 
 * Complex Number Support:
 * - When computeType is 'vec2<f32>', uses complexExpr if available
 * - Complex abs() returns f32 (modulus), requires special handling
 * - Complex comparisons return bool
 */

import type { ITensorIterator } from '@kandle/types';
import type { PointwiseOpConfig, ExtendedPointwiseOpConfig } from './types';
import { generateLoadSnippet } from '../../shader/ShaderSnippets';
import { getComputeType, generateCastSnippet } from '../../base/dtype';
import { getGlobalDTypeResolver } from '../../base/DTypeResolver';

const MAX_RANK = 8;

/**
 * Check if compute type is complex (vec2<f32>)
 */
function isComplexComputeType(computeType: string): boolean {
    return computeType === 'vec2<f32>';
}

/**
 * Generate the operation expression, choosing between scalar and complex
 * @param opConfig The operation configuration
 * @param inputVars Input variable names
 * @param scalarRefs Scalar references
 * @param computeType The WGSL compute type
 * @param dispatchKey The operation dispatch key (for error messages)
 * @returns The WGSL expression string
 */
function generateOpExpression(
    opConfig: PointwiseOpConfig,
    inputVars: string[],
    scalarRefs: Record<string, string>,
    computeType: string,
    dispatchKey: string
): string {
    if (isComplexComputeType(computeType)) {
        // Complex path: use complexExpr if available
        if (opConfig.complexExpr) {
            return opConfig.complexExpr(inputVars, scalarRefs);
        }
        // Fallback: try scalar expr (may work for some ops like copy)
        // But warn in debug that this might not be correct
        throw new Error(
            `Operation '${dispatchKey}' does not support complex numbers (no complexExpr defined). ` +
            `Compute type: ${computeType}`
        );
    }
    // Scalar path: use standard expr
    return opConfig.expr(inputVars, scalarRefs, computeType);
}

export function buildPointwiseShader(
    iter: ITensorIterator,
    dispatchKey: string,
    opConfig: PointwiseOpConfig,
    scalarArgs: Record<string, number>
): string {
    // Choose path based on contiguity
    if (iter.isContiguous) {
        return buildFastPathShader(iter, dispatchKey, opConfig, scalarArgs);
    } else {
        return buildGeneralPathShader(iter, dispatchKey, opConfig, scalarArgs);
    }
}

/**
 * Fast path: All operands are contiguous, use direct gid access
 */
function buildFastPathShader(
    iter: ITensorIterator,
    dispatchKey: string,
    opConfig: PointwiseOpConfig,
    scalarArgs: Record<string, number>
): string {
    const lines: string[] = [];
    const resolver = getGlobalDTypeResolver();
    const computeType = getComputeType(iter.computeDtype);

    // Header comment
    lines.push(`// v5 Pointwise Shader: ${dispatchKey} (FAST PATH)`);
    lines.push(`// Inputs: ${iter.numInputs}, Output shape: [${iter.outputShape.join(', ')}]`);
    lines.push('');

    // Uniforms struct (simple for fast path)
    lines.push('struct Uniforms {');
    lines.push('    numel: u32,');

    // Offsets for each operand (even contiguous tensors may have storage offset)
    for (let i = 0; i < iter.numInputs; i++) {
        lines.push(`    offset_input${i}: u32,`);
    }
    lines.push('    offset_output: u32,');

    // Scalar args (SORTED for consistency)
    const sortedKeys = Object.keys(scalarArgs).sort();
    for (const key of sortedKeys) {
        lines.push(`    ${key}: f32,`);
    }
    lines.push('};');
    lines.push('');

    // Bindings
    lines.push('@group(0) @binding(0) var<uniform> uniforms: Uniforms;');

    for (let i = 0; i < iter.numInputs; i++) {
        const dtype = iter.input(i).dtype;
        const storageType = resolver.getDescriptor(dtype).wgslStorageType;
        lines.push(`@group(0) @binding(${i + 1}) var<storage, read> input${i}: array<${storageType}>;`);
    }

    const outputDtype = iter.output().dtype;
    const outputStorageType = resolver.getDescriptor(outputDtype).wgslStorageType;
    lines.push(`@group(0) @binding(${iter.numInputs + 1}) var<storage, read_write> output: array<${outputStorageType}>;`);
    lines.push('');

    // Generate load helper functions
    const loadHelpers: Array<{ funcName: string; code: string; returnType: string }> = [];
    for (let i = 0; i < iter.numInputs; i++) {
        const inputDtype = iter.input(i).dtype;
        const loadHelper = generateLoadSnippet(`input${i}`, inputDtype);
        loadHelpers.push(loadHelper);
        lines.push(loadHelper.code);
        lines.push('');
    }

    // Generate helper functions if needed (e.g., bessel_i0, sinc_impl)
    if (opConfig.helperFunctions) {
        for (const helperFn of opConfig.helperFunctions) {
            lines.push(helperFn.trim());
            lines.push('');
        }
    }

    // Main function
    // 注意: 使用多维 dispatch 来处理大 tensor (workgroups 可能超过 65535)
    // 全局索引 = (workgroup_id.z * 65535 * 65535 + workgroup_id.y * 65535 + workgroup_id.x) * 64 + local_id
    lines.push('@compute @workgroup_size(64)');
    lines.push('fn main(@builtin(global_invocation_id) globalId: vec3<u32>, @builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {');
    lines.push('    // 计算线性 workgroup 索引 (支持多维 dispatch)');
    lines.push('    let linearWorkgroupId = workgroupId.x + workgroupId.y * 65535u + workgroupId.z * 65535u * 65535u;');
    lines.push('    let gid = linearWorkgroupId * 64u + localId.x;');
    lines.push('    if (gid >= uniforms.numel) { return; }');
    lines.push('');

    // Load inputs with offset
    for (let i = 0; i < iter.numInputs; i++) {
        const inputName = `a${i}`;
        const loadHelper = loadHelpers[i];
        const loadedValue = `${loadHelper.funcName}(gid + uniforms.offset_input${i})`;

        // Cast to compute type if needed
        const finalValue = loadHelper.returnType === computeType
            ? loadedValue
            : generateCastSnippet(loadedValue, loadHelper.returnType as any, computeType as any);

        lines.push(`    let ${inputName} = ${finalValue};`);
    }
    lines.push('');

    // Compute expression - choose scalar or complex based on computeType
    const inputVars = Array.from({ length: iter.numInputs }, (_, i) => `a${i}`);
    const scalarRefs: Record<string, string> = {};
    for (const key of sortedKeys) {
        // For complex types, scalar uniforms stay as f32
        scalarRefs[key] = isComplexComputeType(computeType)
            ? `uniforms.${key}`
            : `${computeType}(uniforms.${key})`;
    }

    const expr = generateOpExpression(opConfig, inputVars, scalarRefs, computeType, dispatchKey);

    // Check if this operation returns bool (comparison operations)
    const isBoolOutput = (opConfig as ExtendedPointwiseOpConfig).outputKind === 'bool';

    if (isBoolOutput) {
        lines.push(`    let boolResult = ${expr};`);
        // For bool output, store as u32 (0 or 1)
        lines.push(`    let result = select(0u, 1u, boolResult);`);
    } else {
        lines.push(`    let result = ${expr};`);
    }
    lines.push('');

    // Store result with offset
    // Note: For bool output, result is already u32 (from select(0u, 1u, boolResult))
    // so we skip the cast and write directly
    if (isBoolOutput) {
        lines.push(`    output[gid + uniforms.offset_output] = result;`);
    } else {
        const outputComputeType = getComputeType(outputDtype);
        const storeValue = computeType === outputComputeType
            ? 'result'
            : generateCastSnippet('result', computeType as any, outputComputeType as any);
        lines.push(`    output[gid + uniforms.offset_output] = ${storeValue};`);
    }

    lines.push('}');
    lines.push('');

    return lines.join('\n');
}

/**
 * General path: Non-contiguous operands, use strided index calculation
 * Uses manually generated offset calculation (not ShaderSnippets.generateOffsetFunction)
 * 
 * Memory Layout for Uniforms (WGSL aligned):
 * - numel: u32 (4 bytes) + padding (12 bytes) = 16 bytes total
 * - shape: array<u32, 8> (32 bytes)
 * - For each input: strides: array<u32, 8> (32 bytes), offset: u32 (4 bytes) + pad (12 bytes)
 * - output: strides: array<u32, 8> (32 bytes), offset: u32 (4 bytes)
 * - rank: u32 (4 bytes) + padding
 * - scalars: f32 each
 */
function buildGeneralPathShader(
    iter: ITensorIterator,
    dispatchKey: string,
    opConfig: PointwiseOpConfig,
    scalarArgs: Record<string, number>
): string {
    const lines: string[] = [];
    const resolver = getGlobalDTypeResolver();
    const computeType = getComputeType(iter.computeDtype);
    const rank = iter.outputShape.length;

    // Header comment
    lines.push(`// v5 Pointwise Shader: ${dispatchKey} (GENERAL PATH - Strided)`);
    lines.push(`// Inputs: ${iter.numInputs}, Output shape: [${iter.outputShape.join(', ')}], Rank: ${rank}`);
    lines.push('');

    // Uniforms struct - use vec4<u32> instead of array for proper alignment
    // This uses a flattened approach that's easier to manage
    lines.push('struct Uniforms {');
    lines.push('    numel: u32,');
    lines.push('    rank: u32,');
    lines.push('    _pad0: u32,');
    lines.push('    _pad1: u32,');

    // Shape: 8 u32 values
    lines.push('    shape0: vec4<u32>,');
    lines.push('    shape1: vec4<u32>,');

    // For each input: strides (8 u32) + offset (1 u32, padded to vec4)
    for (let i = 0; i < iter.numInputs; i++) {
        lines.push(`    strides_input${i}_0: vec4<u32>,`);
        lines.push(`    strides_input${i}_1: vec4<u32>,`);
        lines.push(`    offset_input${i}: u32,`);
        if (i < iter.numInputs - 1 || iter.numInputs === 1) {
            // Need padding after single offset for alignment
            lines.push(`    _pad_i${i}_0: u32,`);
            lines.push(`    _pad_i${i}_1: u32,`);
            lines.push(`    _pad_i${i}_2: u32,`);
        }
    }

    // Output strides (8 u32) + offset
    lines.push('    strides_output_0: vec4<u32>,');
    lines.push('    strides_output_1: vec4<u32>,');
    lines.push('    offset_output: u32,');

    // Scalar args (SORTED) - each f32, padded
    const sortedKeys = Object.keys(scalarArgs).sort();
    if (sortedKeys.length > 0) {
        lines.push('    _pad_out: u32,');
        lines.push('    _pad_out2: u32,');
        lines.push('    _pad_out3: u32,');
        for (const key of sortedKeys) {
            lines.push(`    ${key}: f32,`);
        }
    }
    lines.push('};');
    lines.push('');

    // Bindings
    lines.push('@group(0) @binding(0) var<uniform> uniforms: Uniforms;');

    for (let i = 0; i < iter.numInputs; i++) {
        const dtype = iter.input(i).dtype;
        const storageType = resolver.getDescriptor(dtype).wgslStorageType;
        lines.push(`@group(0) @binding(${i + 1}) var<storage, read> input${i}: array<${storageType}>;`);
    }

    const outputDtype = iter.output().dtype;
    const outputStorageType = resolver.getDescriptor(outputDtype).wgslStorageType;
    lines.push(`@group(0) @binding(${iter.numInputs + 1}) var<storage, read_write> output: array<${outputStorageType}>;`);
    lines.push('');

    // Generate load helper functions
    const loadHelpers: Array<{ funcName: string; code: string; returnType: string }> = [];
    for (let i = 0; i < iter.numInputs; i++) {
        const inputDtype = iter.input(i).dtype;
        const loadHelper = generateLoadSnippet(`input${i}`, inputDtype);
        loadHelpers.push(loadHelper);
        lines.push(loadHelper.code);
        lines.push('');
    }

    // Generate helper functions to access shape and strides
    lines.push(`fn get_shape(dim: u32) -> u32 {
    if (dim < 4u) {
        return uniforms.shape0[dim];
    } else {
        return uniforms.shape1[dim - 4u];
    }
}`);
    lines.push('');

    // Generate offset calculation for each input
    for (let i = 0; i < iter.numInputs; i++) {
        lines.push(`fn get_stride_input${i}(dim: u32) -> u32 {
    if (dim < 4u) {
        return uniforms.strides_input${i}_0[dim];
    } else {
        return uniforms.strides_input${i}_1[dim - 4u];
    }
}`);
        lines.push('');

        lines.push(`fn get_offset_input${i}(idx: u32) -> u32 {
    var offset: u32 = 0u;
    var remaining = idx;
    for (var d: i32 = i32(uniforms.rank) - 1; d >= 0; d = d - 1) {
        let dim = u32(d);
        let dim_size = get_shape(dim);
        let coord = remaining % dim_size;
        remaining = remaining / dim_size;
        offset = offset + coord * get_stride_input${i}(dim);
    }
    return offset;
}`);
        lines.push('');
    }

    // Output offset function
    lines.push(`fn get_stride_output(dim: u32) -> u32 {
    if (dim < 4u) {
        return uniforms.strides_output_0[dim];
    } else {
        return uniforms.strides_output_1[dim - 4u];
    }
}`);
    lines.push('');

    lines.push(`fn get_offset_output(idx: u32) -> u32 {
    var offset: u32 = 0u;
    var remaining = idx;
    for (var d: i32 = i32(uniforms.rank) - 1; d >= 0; d = d - 1) {
        let dim = u32(d);
        let dim_size = get_shape(dim);
        let coord = remaining % dim_size;
        remaining = remaining / dim_size;
        offset = offset + coord * get_stride_output(dim);
    }
    return offset;
}`);
    lines.push('');

    // Generate helper functions if needed (e.g., bessel_i0, sinc_impl)
    if (opConfig.helperFunctions) {
        for (const helperFn of opConfig.helperFunctions) {
            lines.push(helperFn.trim());
            lines.push('');
        }
    }

    // Main function
    // 注意: 使用多维 dispatch 来处理大 tensor (workgroups 可能超过 65535)
    // 全局索引 = (workgroup_id.z * 65535 * 65535 + workgroup_id.y * 65535 + workgroup_id.x) * 64 + local_id
    lines.push('@compute @workgroup_size(64)');
    lines.push('fn main(@builtin(global_invocation_id) globalId: vec3<u32>, @builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {');
    lines.push('    // 计算线性 workgroup 索引 (支持多维 dispatch)');
    lines.push('    let linearWorkgroupId = workgroupId.x + workgroupId.y * 65535u + workgroupId.z * 65535u * 65535u;');
    lines.push('    let gid = linearWorkgroupId * 64u + localId.x;');
    lines.push('    if (gid >= uniforms.numel) { return; }');
    lines.push('');

    // Load inputs using strided offset calculation
    for (let i = 0; i < iter.numInputs; i++) {
        const inputName = `a${i}`;
        const loadHelper = loadHelpers[i];
        const offsetCall = `get_offset_input${i}(gid) + uniforms.offset_input${i}`;
        const loadedValue = `${loadHelper.funcName}(${offsetCall})`;

        // Cast to compute type if needed
        const finalValue = loadHelper.returnType === computeType
            ? loadedValue
            : generateCastSnippet(loadedValue, loadHelper.returnType as any, computeType as any);

        lines.push(`    let ${inputName} = ${finalValue};`);
    }
    lines.push('');

    // Compute expression - choose scalar or complex based on computeType
    const inputVars = Array.from({ length: iter.numInputs }, (_, i) => `a${i}`);
    const scalarRefs: Record<string, string> = {};
    for (const key of sortedKeys) {
        // For complex types, scalar uniforms stay as f32
        scalarRefs[key] = isComplexComputeType(computeType)
            ? `uniforms.${key}`
            : `${computeType}(uniforms.${key})`;
    }

    const expr = generateOpExpression(opConfig, inputVars, scalarRefs, computeType, dispatchKey);

    // Check if this operation returns bool
    const isBoolOutput = (opConfig as ExtendedPointwiseOpConfig).outputKind === 'bool';

    if (isBoolOutput) {
        lines.push(`    let boolResult = ${expr};`);
        // For bool output, store as u32 (0 or 1)
        lines.push(`    let result = select(0u, 1u, boolResult);`);
    } else {
        lines.push(`    let result = ${expr};`);
    }
    lines.push('');

    // Store result using strided offset
    // Note: For bool output, result is already u32 (from select(0u, 1u, boolResult))
    // so we skip the cast and write directly
    const outputOffsetCall = `get_offset_output(gid) + uniforms.offset_output`;
    if (isBoolOutput) {
        lines.push(`    output[${outputOffsetCall}] = result;`);
    } else {
        const outputComputeType = getComputeType(outputDtype);
        const storeValue = computeType === outputComputeType
            ? 'result'
            : generateCastSnippet('result', computeType as any, outputComputeType as any);
        lines.push(`    output[${outputOffsetCall}] = ${storeValue};`);
    }

    lines.push('}');
    lines.push('');

    return lines.join('\n');
}
