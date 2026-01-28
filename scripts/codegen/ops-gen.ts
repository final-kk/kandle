/**
 * v5 CodeGen - Ops Generator
 *
 * 生成 ops.ts - API 入口点
 * 使用编译时确定的 typeof 分支选择正确的变体
 */

import type { OpEntry, ValueType, ParamDef, ReturnDef } from '../../packages/types/src/opschema/types';
import { getInternalFuncName } from './internal-gen';

type GroupedEntries = Map<string, OpEntry[]>;

/**
 * Generate ops.ts content
 */
export function generateOps(grouped: GroupedEntries): string {
    const lines: string[] = [];

    // Header
    lines.push('/**');
    lines.push(' * v5 Generated Operators');
    lines.push(' * DO NOT EDIT - Generated from OpRegistry');

    lines.push(' *');
    lines.push(' * Each operator uses typeof branching to dispatch to the correct variant');
    lines.push(' */');
    lines.push('');

    // Imports
    lines.push("import type { ITensorHandle, DType } from '@kandle/types';");
    lines.push("import { Tensor } from '../tensor';");
    lines.push("import * as internal from './internal';");
    lines.push('');

    // Generate each operator
    lines.push('// ============================================================================');
    lines.push('// Operators');
    lines.push('// ============================================================================');
    lines.push('');

    for (const [opName, entries] of grouped) {
        lines.push(generateOperator(opName, entries));
        lines.push('');
    }

    // ========================================================================
    // Generate nn.functional namespace object
    // Collect all ops with namespace: 'nn.functional'
    // ========================================================================
    const nnFunctionalOps: string[] = [];
    for (const [opName, entries] of grouped) {
        // Check if any entry has namespace: 'nn.functional'
        const hasNnFunctional = entries.some(e => e.codegen && typeof e.codegen === 'object' && e.codegen.namespace === 'nn.functional');
        if (hasNnFunctional) {
            nnFunctionalOps.push(opName);
        }
    }

    if (nnFunctionalOps.length > 0) {
        // Sort for consistent output
        nnFunctionalOps.sort();

        lines.push('// ============================================================================');
        lines.push('// nn.functional namespace');
        lines.push('// ============================================================================');
        lines.push('');
        lines.push('/**');
        lines.push(' * Functional interface for neural network operations');
        lines.push(' * Provides PyTorch-compatible nn.functional.* API');
        lines.push(' * @see https://pytorch.org/docs/stable/nn.functional.html');
        lines.push(' */');
        lines.push('export const functional = {');
        for (const opName of nnFunctionalOps) {
            lines.push(`    ${opName},`);
        }
        lines.push('} as const;');
        lines.push('');
    }

    // ========================================================================
    // Generate fft namespace object
    // Collect all ops with namespace: 'fft'
    // Handle namespaceKeyAlias for operators like fft.fft()
    // ========================================================================
    const fftOps: Array<{ opName: string; alias?: string; funcName: string }> = [];
    for (const [opName, entries] of grouped) {
        // Check if any entry has namespace: 'fft'
        const fftEntry = entries.find(e => e.codegen && typeof e.codegen === 'object' && e.codegen.namespace === 'fft');
        if (fftEntry) {
            const alias = fftEntry.codegen?.namespaceKeyAlias;
            const funcName = alias ? `${opName}Impl` : opName;
            fftOps.push({ opName, alias, funcName });
        }
    }

    if (fftOps.length > 0) {
        // Sort by alias or opName for consistent output
        fftOps.sort((a, b) => (a.alias || a.opName).localeCompare(b.alias || b.opName));

        lines.push('// ============================================================================');
        lines.push('// fft namespace (torch.fft.*)');
        lines.push('// ============================================================================');
        lines.push('');
        lines.push('/**');
        lines.push(' * FFT operations namespace');
        lines.push(' * Provides PyTorch-compatible torch.fft.* API');
        lines.push(' * @see https://pytorch.org/docs/stable/fft.html');
        lines.push(' */');
        lines.push('export const fft = {');
        for (const op of fftOps) {
            if (op.alias) {
                // Use alias as key, funcName (with Impl suffix) as value
                lines.push(`    ${op.alias}: ${op.funcName},`);
            } else {
                // No alias, use opName directly
                lines.push(`    ${op.opName},`);
            }
        }
        lines.push('} as const;');
        lines.push('');
    }

    return lines.join('\n');
}

/**
 * Generate a single operator with all its variants
 */
function generateOperator(opName: string, entries: OpEntry[]): string {
    const lines: string[] = [];

    // Skip codegen if disabled
    if (entries[0].codegen === false as any) {
        return '';
    }

    // Check for namespaceKeyAlias - if set, function name becomes {name}Impl
    const namespaceKeyAlias = entries[0].codegen?.namespaceKeyAlias;
    const actualFuncName = namespaceKeyAlias ? `${opName}Impl` : opName;

    // Check if operator has variants
    const hasVariants = entries.length > 1 || entries[0].variant !== undefined;

    if (!hasVariants) {
        // Single variant - simple delegation
        lines.push(generateSingleVariant(actualFuncName, opName, entries[0]));
    } else {
        // Multiple variants - generate overloads + typeof dispatch
        lines.push(generateMultiVariant(actualFuncName, opName, entries));
    }

    return lines.join('\n');
}

/**
 * Generate a single-variant operator
 * @param funcName - The function name to generate (may include Impl suffix)
 * @param opName - The original operator name (for internal function lookup)
 */
function generateSingleVariant(funcName: string, opName: string, entry: OpEntry): string {
    const lines: string[] = [];
    const internalFuncName = getInternalFuncName(entry);

    // Check for conditional return
    const condRet = entry.codegen?.conditionalReturn;

    if (condRet) {
        // Generate with conditional return handling
        return generateConditionalReturnOperator(funcName, entry, internalFuncName, condRet);
    }

    // Doc comment (add note if function has Impl suffix)
    const implNote = funcName !== opName ? ` (internal: ${funcName} to avoid namespace conflict)` : '';
    if (entry.doc) {
        lines.push(`/** ${entry.doc}${implNote} */`);
    }

    // Function signature
    const params = generateApiParams(entry.signature.params);
    const returnType = generateApiReturnType(entry.signature.returns);

    lines.push(`export function ${funcName}(`);
    for (let i = 0; i < params.length; i++) {
        const comma = i < params.length - 1 ? ',' : '';
        lines.push(`    ${params[i]}${comma}`);
    }
    lines.push(`): ${returnType} {`);

    // Body - call internal and wrap result
    const argList = entry.signature.params.map(p => convertArgToHandle(p)).join(', ');
    const isTuple = 'tuple' in entry.signature.returns;

    if (isTuple) {
        const tupleSize = entry.signature.returns.tuple.length;
        lines.push(`    const result = internal.${internalFuncName}(${argList});`);
        lines.push(`    return result.map(h => new Tensor(h)) as [${Array(tupleSize).fill('Tensor').join(', ')}];`);
    } else {
        lines.push(`    const result = internal.${internalFuncName}(${argList});`);
        lines.push(`    return new Tensor(result);`);
    }

    lines.push('}');

    return lines.join('\n');
}

/**
 * Generate operator with conditional return type
 * (e.g., maxPool2d returns Tensor normally, [Tensor, Tensor] when returnIndices=true)
 */
function generateConditionalReturnOperator(
    opName: string,
    entry: OpEntry,
    funcName: string,
    condRet: { param: string; tupleSize: number }
): string {
    const lines: string[] = [];
    const params = generateApiParams(entry.signature.params);
    const argList = entry.signature.params.map(p => convertArgToHandle(p)).join(', ');
    const tupleType = `[${Array(condRet.tupleSize).fill('Tensor').join(', ')}]`;

    // Doc comment
    if (entry.doc) {
        lines.push(`/** ${entry.doc} */`);
    }

    // Overload 1: param=false (or undefined) -> single return
    lines.push(`export function ${opName}(`);
    for (let i = 0; i < params.length; i++) {
        const p = params[i];
        // Replace the conditional param with literal false
        const modified = p.includes(`${condRet.param}?:`)
            ? p.replace(`${condRet.param}?:`, `${condRet.param}?:`)
            : p;
        const comma = i < params.length - 1 ? ',' : '';
        lines.push(`    ${modified}${comma}`);
    }
    lines.push(`): Tensor;`);

    // Overload 2: param=true -> tuple return  
    lines.push(`export function ${opName}(`);
    for (let i = 0; i < params.length; i++) {
        const p = params[i];
        const comma = i < params.length - 1 ? ',' : '';
        lines.push(`    ${p}${comma}`);
    }
    lines.push(`): ${tupleType};`);

    // Implementation
    lines.push(`export function ${opName}(`);
    for (let i = 0; i < params.length; i++) {
        const comma = i < params.length - 1 ? ',' : '';
        lines.push(`    ${params[i]}${comma}`);
    }
    lines.push(`): Tensor | ${tupleType} {`);
    lines.push(`    const result = internal.${funcName}(${argList});`);
    lines.push(`    // Handle conditional return based on ${condRet.param}`);
    lines.push(`    if (Array.isArray(result)) {`);
    lines.push(`        return result.map(h => new Tensor(h)) as ${tupleType};`);
    lines.push(`    }`);
    lines.push(`    return new Tensor(result);`);
    lines.push('}');

    return lines.join('\n');
}

/**
 * Generate a multi-variant operator with typeof dispatch
 * @param funcName - The function name to generate (may include Impl suffix)
 * @param opName - The original operator name (for internal function lookup)
 */
function generateMultiVariant(funcName: string, opName: string, entries: OpEntry[]): string {
    const lines: string[] = [];

    // Find the parameter that differs between variants (usually 'other' or second param)
    const variantParam = findVariantParameter(entries);
    // Find parameter that distinguishes by presence (e.g., dim in max/min)
    const presenceParam = findPresenceParameter(entries);

    // Generate overload declarations
    for (const entry of entries) {
        const variantSuffix = entry.variant ? ` (${entry.variant} variant)` : '';
        if (entry.doc) {
            lines.push(`/** ${entry.doc}${variantSuffix} */`);
        }
        const params = generateApiParams(entry.signature.params);
        const returnType = generateApiReturnType(entry.signature.returns);

        lines.push(`export function ${funcName}(${params.join(', ')}): ${returnType};`);
    }

    // Generate unified implementation
    const unifiedParams = buildUnifiedParams(entries);
    lines.push(`export function ${funcName}(`);
    for (let i = 0; i < unifiedParams.length; i++) {
        const comma = i < unifiedParams.length - 1 ? ',' : '';
        lines.push(`    ${unifiedParams[i]}${comma}`);
    }
    lines.push('): Tensor | [Tensor, Tensor] {');

    // Generate dispatch logic
    if (variantParam) {
        // Type-based dispatch (Tensor vs Scalar)
        lines.push(generateTypeofDispatch(opName, entries, variantParam));
    } else if (presenceParam) {
        // Presence-based dispatch (e.g., max(self) vs max(self, dim))
        lines.push(generatePresenceDispatch(opName, entries, presenceParam));
    } else {
        // Fallback: call first variant
        const firstEntry = entries[0];
        const funcName = getInternalFuncName(firstEntry);
        const argList = firstEntry.signature.params.map(p => convertArgToHandle(p)).join(', ');
        const isTuple = 'tuple' in firstEntry.signature.returns;
        if (isTuple) {
            lines.push(`    const result = internal.${funcName}(${argList});`);
            lines.push(`    return result.map(h => new Tensor(h)) as [Tensor, Tensor];`);
        } else {
            lines.push(`    const result = internal.${funcName}(${argList});`);
            lines.push(`    return new Tensor(result);`);
        }
    }

    lines.push('}');

    return lines.join('\n');
}

/**
 * Find the parameter that distinguishes variants (Tensor in one, Scalar in another)
 */
function findVariantParameter(entries: OpEntry[]): string | null {
    const paramTypes = new Map<string, Set<string>>();

    for (const entry of entries) {
        for (const param of entry.signature.params) {
            if (!paramTypes.has(param.name)) {
                paramTypes.set(param.name, new Set());
            }
            paramTypes.get(param.name)!.add(getBaseKind(param.type));
        }
    }

    // Find param with both Tensor and Scalar
    for (const [name, kinds] of paramTypes) {
        if (kinds.has('Tensor') && kinds.has('Scalar')) {
            return name;
        }
    }

    return null;
}

/**
 * Find a parameter that exists in some variants but not others
 * (for dispatch based on parameter presence, e.g. max(self) vs max(self, dim))
 */
function findPresenceParameter(entries: OpEntry[]): string | null {
    if (entries.length < 2) return null;

    // Get all param names from each variant
    const paramSets = entries.map(e => new Set(e.signature.params.map(p => p.name)));

    // Find params that appear in some but not all variants
    const allParams = new Set<string>();
    for (const ps of paramSets) {
        for (const p of ps) {
            allParams.add(p);
        }
    }

    for (const param of allParams) {
        const appearanceCount = paramSets.filter(ps => ps.has(param)).length;
        if (appearanceCount > 0 && appearanceCount < entries.length) {
            return param;
        }
    }

    return null;
}

/**
 * Get the base kind of a ValueType (unwrap Optional)
 */
function getBaseKind(type: ValueType): string {
    if (type.kind === 'Optional') {
        return type.inner.kind;
    }
    return type.kind;
}

/**
 * Generate typeof dispatch code
 */
function generateTypeofDispatch(opName: string, entries: OpEntry[], variantParam: string): string {
    const lines: string[] = [];

    // Find Tensor and Scalar variants
    const tensorEntry = entries.find(e => {
        const param = e.signature.params.find(p => p.name === variantParam);
        return param && getBaseKind(param.type) === 'Tensor';
    });

    const scalarEntry = entries.find(e => {
        const param = e.signature.params.find(p => p.name === variantParam);
        return param && getBaseKind(param.type) === 'Scalar';
    });

    if (!tensorEntry || !scalarEntry) {
        // Fallback
        const firstEntry = entries[0];
        const funcName = getInternalFuncName(firstEntry);
        lines.push(`    const result = internal.${funcName}(${firstEntry.signature.params.map(p => convertArgToHandle(p)).join(', ')});`);
        lines.push(`    return new Tensor(result);`);
        return lines.join('\n');
    }

    // typeof check
    lines.push(`    if (typeof ${variantParam} === 'number') {`);

    // Scalar variant
    const scalarFuncName = getInternalFuncName(scalarEntry);
    const scalarArgs = scalarEntry.signature.params.map(p => convertArgToHandle(p)).join(', ');
    const scalarIsTuple = 'tuple' in scalarEntry.signature.returns;

    if (scalarIsTuple) {
        lines.push(`        const result = internal.${scalarFuncName}(${scalarArgs});`);
        lines.push(`        return result.map(h => new Tensor(h)) as [Tensor, Tensor];`);
    } else {
        lines.push(`        const result = internal.${scalarFuncName}(${scalarArgs});`);
        lines.push(`        return new Tensor(result);`);
    }

    lines.push('    } else {');

    // Tensor variant
    const tensorFuncName = getInternalFuncName(tensorEntry);
    const tensorArgs = tensorEntry.signature.params.map(p => convertArgToHandle(p, variantParam)).join(', ');
    const tensorIsTuple = 'tuple' in tensorEntry.signature.returns;

    if (tensorIsTuple) {
        lines.push(`        const result = internal.${tensorFuncName}(${tensorArgs});`);
        lines.push(`        return result.map(h => new Tensor(h)) as [Tensor, Tensor];`);
    } else {
        lines.push(`        const result = internal.${tensorFuncName}(${tensorArgs});`);
        lines.push(`        return new Tensor(result);`);
    }

    lines.push('    }');

    return lines.join('\n');
}

/**
 * Generate dispatch based on parameter presence
 * (e.g., max(self) vs max(self, dim, keepdim))
 */
function generatePresenceDispatch(opName: string, entries: OpEntry[], presenceParam: string): string {
    const lines: string[] = [];

    // Find entry without the param (fewer params) and entry with the param
    const withoutParam = entries.find(e =>
        !e.signature.params.some(p => p.name === presenceParam)
    );
    const withParam = entries.find(e =>
        e.signature.params.some(p => p.name === presenceParam)
    );

    if (!withoutParam || !withParam) {
        // Fallback to first entry
        const firstEntry = entries[0];
        const funcName = getInternalFuncName(firstEntry);
        const argList = firstEntry.signature.params.map(p => convertArgToHandle(p)).join(', ');
        lines.push(`    const result = internal.${funcName}(${argList});`);
        lines.push(`    return new Tensor(result);`);
        return lines.join('\n');
    }

    // Check if param is provided (not undefined)
    lines.push(`    if (${presenceParam} !== undefined) {`);

    // With param variant
    const withFuncName = getInternalFuncName(withParam);
    const withArgs = withParam.signature.params.map(p => convertArgToHandle(p)).join(', ');
    const withIsTuple = 'tuple' in withParam.signature.returns;

    if (withIsTuple) {
        lines.push(`        const result = internal.${withFuncName}(${withArgs});`);
        lines.push(`        return result.map(h => new Tensor(h)) as [Tensor, Tensor];`);
    } else {
        lines.push(`        const result = internal.${withFuncName}(${withArgs});`);
        lines.push(`        return new Tensor(result);`);
    }

    lines.push('    } else {');

    // Without param variant
    const withoutFuncName = getInternalFuncName(withoutParam);
    const withoutArgs = withoutParam.signature.params.map(p => convertArgToHandle(p)).join(', ');
    const withoutIsTuple = 'tuple' in withoutParam.signature.returns;

    if (withoutIsTuple) {
        lines.push(`        const result = internal.${withoutFuncName}(${withoutArgs});`);
        lines.push(`        return result.map(h => new Tensor(h)) as [Tensor, Tensor];`);
    } else {
        lines.push(`        const result = internal.${withoutFuncName}(${withoutArgs});`);
        lines.push(`        return new Tensor(result);`);
    }

    lines.push('    }');

    return lines.join('\n');
}

/**
 * Build unified params for multi-variant operator
 */
function buildUnifiedParams(entries: OpEntry[]): string[] {
    const paramMap = new Map<string, { type: string; optional: boolean; order: number }>();
    let order = 0;

    for (const entry of entries) {
        for (const param of entry.signature.params) {
            if (!paramMap.has(param.name)) {
                paramMap.set(param.name, {
                    type: valueTypeToAPI(param.type),
                    optional: param.default !== undefined || param.type.kind === 'Optional',
                    order: order++,
                });
            } else {
                const existing = paramMap.get(param.name)!;
                // Merge types if different
                const newType = valueTypeToAPI(param.type);
                if (existing.type !== newType && !existing.type.includes(newType)) {
                    existing.type = `${existing.type} | ${newType}`;
                }
                if (param.default !== undefined || param.type.kind === 'Optional') {
                    existing.optional = true;
                }
            }
        }
    }

    // Check for params that don't appear in all signatures
    const sigCount = entries.length;
    for (const [name, info] of paramMap) {
        let count = 0;
        for (const entry of entries) {
            if (entry.signature.params.some(p => p.name === name)) {
                count++;
            }
        }
        if (count < sigCount) {
            info.optional = true;
        }
    }

    // Sort by order and split required/optional
    const sorted = [...paramMap.entries()].sort((a, b) => a[1].order - b[1].order);
    const required = sorted.filter(([, info]) => !info.optional);
    const optional = sorted.filter(([, info]) => info.optional);

    const result: string[] = [];
    for (const [name, info] of [...required, ...optional]) {
        const opt = info.optional ? '?' : '';
        result.push(`${name}${opt}: ${info.type}`);
    }

    return result;
}

/**
 * Generate API parameter signature
 */
function generateApiParams(params: readonly ParamDef[]): string[] {
    // Split into required and optional, required first
    const required = params.filter(p => p.default === undefined && p.type.kind !== 'Optional');
    const optional = params.filter(p => p.default !== undefined || p.type.kind === 'Optional');

    const result: string[] = [];

    for (const p of required) {
        result.push(`${p.name}: ${valueTypeToAPI(p.type)}`);
    }

    for (const p of optional) {
        result.push(`${p.name}?: ${valueTypeToAPI(p.type)}`);
    }

    return result;
}

/**
 * Generate API return type
 */
function generateApiReturnType(returns: ReturnDef): string {
    if ('single' in returns) {
        const type = returns.single;
        if (type.kind === 'Tensor') {
            return 'Tensor';
        } else if (type.kind === 'TensorList') {
            return 'Tensor[]';
        }
        return valueTypeToAPI(type);
    } else {
        // Tuple
        const types = returns.tuple.map(t => valueTypeToAPI(t.type));
        return `[${types.join(', ')}]`;
    }
}

/**
 * Convert ValueType to API-level TypeScript type (uses Tensor, not ITensorHandle)
 */
function valueTypeToAPI(type: ValueType): string {
    switch (type.kind) {
        case 'Tensor':
            return 'Tensor';
        case 'TensorList':
            return 'Tensor[]';
        case 'Scalar':
            return type.numericKind === 'bool' ? 'boolean' : 'number';
        case 'ScalarList':
            return 'number[]';
        case 'Shape':
            return 'number[]';
        case 'Axis':
            return 'number';
        case 'Axes':
            return 'number | number[]';
        case 'DType':
            return 'DType';
        case 'Device':
            return 'string';
        case 'Bool':
            return 'boolean';
        case 'String':
            if (type.oneOf && type.oneOf.length > 0) {
                return type.oneOf.map(s => `'${s}'`).join(' | ');
            }
            return 'string';
        case 'Optional':
            return valueTypeToAPI(type.inner);
        case 'Union':
            return type.types.map(valueTypeToAPI).join(' | ');
        default:
            return 'unknown';
    }
}

/**
 * Convert parameter to handle access for internal function call
 */
function convertArgToHandle(param: ParamDef, forceTensorParam?: string): string {
    const name = param.name;
    const type = param.type;
    const isOpt = type.kind === 'Optional';
    const inner = isOpt ? type.inner : type;

    // If this is the variant parameter and we're in the Tensor branch
    if (forceTensorParam && name === forceTensorParam) {
        if (isOpt) {
            return `${name}?._handle`;
        }
        return `(${name} as Tensor)._handle`;
    }

    switch (inner.kind) {
        case 'Tensor':
            if (isOpt) {
                return `${name}?._handle`;
            }
            return `${name}._handle`;
        case 'TensorList':
            if (isOpt) {
                return `${name}?.map(t => t._handle)`;
            }
            return `${name}.map(t => t._handle)`;
        case 'Union':
            const hasTensor = inner.types.some(t => t.kind === 'Tensor');
            if (hasTensor) {
                // If union contains Tensor, we need to check if it is a Tensor and extract handle
                return `(${name} instanceof Tensor ? ${name}._handle : ${name})`;
            }
            return name;
        default:
            return name;
    }
}
