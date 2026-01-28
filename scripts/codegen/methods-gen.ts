/**
 * v5 CodeGen - Tensor Methods Generator
 *
 * 生成 tensor-methods.ts - Tensor.prototype 方法绑定
 */

import type { OpEntry, ValueType, ParamDef, ReturnDef } from '../../packages/types/src/opschema/types';

type GroupedEntries = Map<string, OpEntry[]>;

/**
 * Generate tensor-methods.ts content
 */
export function generateTensorMethods(grouped: GroupedEntries): string {
    const lines: string[] = [];

    // Header
    lines.push('/**');
    lines.push(' * v5 Generated Tensor Methods');
    lines.push(' * DO NOT EDIT - Generated from OpRegistry');

    lines.push(' */');
    lines.push('');

    // Imports
    lines.push("import { DType } from '@kandle/types';");
    lines.push("import { Tensor } from '../tensor';");
    lines.push("import * as ops from './ops';");
    lines.push('');

    // Collect methods
    const instanceMethods: { methodName: string; opName: string; entries: OpEntry[] }[] = [];
    const staticMethods: { methodName: string; opName: string; entries: OpEntry[] }[] = [];

    for (const [opName, entries] of grouped) {
        const codegen = entries[0].codegen;
        if (codegen === false as any || codegen?.tensorMethod === false) {
            continue;
        }

        const methodName = typeof codegen?.tensorMethod === 'string' ? codegen.tensorMethod : opName;

        if (codegen?.staticMethod) {
            staticMethods.push({ methodName, opName, entries });
        } else {
            instanceMethods.push({ methodName, opName, entries });
        }
    }

    // Module Augmentation
    lines.push('// ============================================================================');
    lines.push('// TypeScript Module Augmentation');
    lines.push('// ============================================================================');
    lines.push('');
    lines.push("declare module '../tensor' {");
    lines.push('    interface Tensor<T extends DType = DType> {');

    for (const { methodName, entries } of instanceMethods) {
        lines.push(generateMethodDeclaration(methodName, entries));
    }

    lines.push('    }');

    // Static methods namespace
    if (staticMethods.length > 0) {
        lines.push('');
        lines.push('    // Static methods');
        lines.push('    namespace Tensor {');
        for (const { methodName, entries } of staticMethods) {
            lines.push(generateStaticMethodDeclaration(methodName, entries));
        }
        lines.push('    }');
    }

    lines.push('}');
    lines.push('');

    // Runtime binding
    lines.push('// ============================================================================');
    lines.push('// Runtime Binding');
    lines.push('// ============================================================================');
    lines.push('');

    for (const { methodName, opName, entries } of instanceMethods) {
        lines.push(generateMethodBinding(methodName, opName, entries));
        lines.push('');
    }

    for (const { methodName, opName, entries } of staticMethods) {
        lines.push(generateStaticMethodBinding(methodName, opName, entries));
        lines.push('');
    }

    return lines.join('\n');
}

/**
 * Generate method declaration for interface augmentation
 */
function generateMethodDeclaration(methodName: string, entries: OpEntry[]): string {
    const lines: string[] = [];

    // Use first entry for doc
    const doc = entries[0].doc;
    const thisArg = entries[0].codegen?.thisArg ?? findFirstTensorParam(entries[0]);

    // Build unified params (excluding this)
    const allParams = new Map<string, { type: string; optional: boolean }>();

    for (const entry of entries) {
        for (const param of entry.signature.params) {
            if (param.name === thisArg) continue;

            const key = param.name;
            if (!allParams.has(key)) {
                allParams.set(key, {
                    type: valueTypeToAPI(param.type),
                    optional: param.default !== undefined || param.type.kind === 'Optional',
                });
            } else {
                const existing = allParams.get(key)!;
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

    // Check for params missing from some signatures
    for (const [name, info] of allParams) {
        let count = 0;
        for (const entry of entries) {
            if (entry.signature.params.some(p => p.name === name && p.name !== thisArg)) {
                count++;
            }
        }
        if (count < entries.length) {
            info.optional = true;
        }
    }

    // Build return type union
    const returnTypes = new Set<string>();
    for (const entry of entries) {
        returnTypes.add(generateApiReturnType(entry.signature.returns));
    }

    const returnType = Array.from(returnTypes).join(' | ');
    const paramList = buildParamList(allParams);

    if (doc) {
        lines.push(`        /** ${doc} */`);
    }
    lines.push(`        ${methodName}(${paramList}): ${returnType};`);

    return lines.join('\n');
}

/**
 * Generate static method declaration
 */
function generateStaticMethodDeclaration(methodName: string, entries: OpEntry[]): string {
    const lines: string[] = [];
    const entry = entries[0];
    const doc = entry.doc;

    const params = entry.signature.params.map(p => {
        const opt = p.default !== undefined || p.type.kind === 'Optional' ? '?' : '';
        return `${p.name}${opt}: ${valueTypeToAPI(p.type)}`;
    }).join(', ');

    const returnType = generateApiReturnType(entry.signature.returns);

    if (doc) {
        lines.push(`        /** ${doc} */`);
    }
    lines.push(`        function ${methodName}(${params}): ${returnType};`);

    return lines.join('\n');
}

/**
 * Generate runtime method binding
 */
function generateMethodBinding(methodName: string, opName: string, entries: OpEntry[]): string {
    const lines: string[] = [];
    const thisArg = entries[0].codegen?.thisArg ?? findFirstTensorParam(entries[0]);

    // Collect all non-this params
    const allParams = new Set<string>();
    for (const entry of entries) {
        for (const param of entry.signature.params) {
            if (param.name !== thisArg) {
                allParams.add(param.name);
            }
        }
    }

    const paramList = Array.from(allParams).map(p => `${p}: any`).join(', ');

    // Build call args (this goes in place of thisArg)
    const firstSig = entries[0].signature;
    const callArgs = firstSig.params.map(p => {
        if (p.name === thisArg) return 'this';
        return p.name;
    }).join(', ');

    lines.push(`Tensor.prototype.${methodName} = function(${paramList}) {`);
    lines.push(`    return ops.${opName}(${callArgs});`);
    lines.push('};');

    return lines.join('\n');
}

/**
 * Generate static method binding
 */
function generateStaticMethodBinding(methodName: string, opName: string, entries: OpEntry[]): string {
    const lines: string[] = [];
    const entry = entries[0];

    const paramList = entry.signature.params.map(p => `${p.name}: any`).join(', ');
    const callArgs = entry.signature.params.map(p => p.name).join(', ');

    lines.push(`(Tensor as any).${methodName} = function(${paramList}) {`);
    lines.push(`    return ops.${opName}(${callArgs});`);
    lines.push('};');

    return lines.join('\n');
}

/**
 * Find the first Tensor parameter (typically 'self' or 'input')
 */
function findFirstTensorParam(entry: OpEntry): string {
    for (const param of entry.signature.params) {
        const type = param.type.kind === 'Optional' ? param.type.inner : param.type;
        if (type.kind === 'Tensor') {
            return param.name;
        }
    }
    return 'self';
}

/**
 * Build param list string from map
 */
function buildParamList(params: Map<string, { type: string; optional: boolean }>): string {
    const required: string[] = [];
    const optional: string[] = [];

    for (const [name, info] of params) {
        if (info.optional) {
            optional.push(`${name}?: ${info.type}`);
        } else {
            required.push(`${name}: ${info.type}`);
        }
    }

    return [...required, ...optional].join(', ');
}

/**
 * Convert ValueType to API-level TypeScript type
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
        const types = returns.tuple.map(t => 'Tensor');
        return `[${types.join(', ')}]`;
    }
}
