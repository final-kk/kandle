/**
 * v5 CodeGen - Types Generator
 *
 * 生成 types.d.ts - TypeScript 类型声明
 */

import type { OpEntry, ValueType, ParamDef, ReturnDef } from '../../packages/types/src/opschema/types';

type GroupedEntries = Map<string, OpEntry[]>;

/**
 * Generate types.d.ts content
 */
export function generateTypes(grouped: GroupedEntries): string {
    const lines: string[] = [];

    // Header
    lines.push('/**');
    lines.push(' * v5 Generated Type Declarations');
    lines.push(' * DO NOT EDIT - Generated from OpRegistry');

    lines.push(' */');
    lines.push('');

    lines.push("import type { DType } from '@kandle/types';");
    lines.push("import type { Tensor } from '../tensor';");
    lines.push('');

    // Export all operator function types
    lines.push('// Operator function signatures');
    lines.push('export interface Ops {');

    for (const [opName, entries] of grouped) {
        lines.push(generateOpTypeDeclaration(opName, entries));
    }

    lines.push('}');
    lines.push('');

    // Export namespace for nn.functional
    const nnFunctional = [...grouped.entries()]
        .filter(([, entries]) => entries[0].codegen?.namespace === 'nn.functional');

    if (nnFunctional.length > 0) {
        lines.push('// nn.functional namespace');
        lines.push('export interface NNFunctional {');
        for (const [opName, entries] of nnFunctional) {
            lines.push(generateOpTypeDeclaration(opName, entries));
        }
        lines.push('}');
        lines.push('');
    }

    return lines.join('\n');
}

/**
 * Generate type declaration for a single operator
 */
function generateOpTypeDeclaration(opName: string, entries: OpEntry[]): string {
    const lines: string[] = [];

    // If multiple variants, use overloaded function type
    if (entries.length > 1) {
        lines.push(`    ${opName}: {`);
        for (const entry of entries) {
            const params = entry.signature.params.map(p => {
                const opt = p.default !== undefined || p.type.kind === 'Optional' ? '?' : '';
                return `${p.name}${opt}: ${valueTypeToAPI(p.type)}`;
            }).join(', ');
            const returnType = generateApiReturnType(entry.signature.returns);
            lines.push(`        (${params}): ${returnType};`);
        }
        lines.push(`    };`);
    } else {
        const entry = entries[0];
        const params = entry.signature.params.map(p => {
            const opt = p.default !== undefined || p.type.kind === 'Optional' ? '?' : '';
            return `${p.name}${opt}: ${valueTypeToAPI(p.type)}`;
        }).join(', ');
        const returnType = generateApiReturnType(entry.signature.returns);
        lines.push(`    ${opName}(${params}): ${returnType};`);
    }

    return lines.join('\n');
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
