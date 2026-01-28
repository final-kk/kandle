/**
 * NN-Kit v5 CodeGen - Main Entry
 *
 * v5 架构的代码生成器：
 * - 每个变体生成独立的 internal/*.ts 实现
 * - ops.ts 使用 typeof 分支选择变体
 * - tensor-methods.ts 生成 Tensor.prototype 方法
 *
 * Run: npx tsx scripts/codegen/v5
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { OpRegistry } from '../../packages/types/src/opschema/registry';
import { generateInternalFiles } from "./internal-gen";
import { generateOps } from './ops-gen';
import { generateTensorMethods } from './methods-gen';
import { generateTypes } from './types-gen';

// ============================================================================
// Configuration
// ============================================================================

// ESM-compatible __dirname
const __filename_esm = fileURLToPath(import.meta.url);
const __dirname_esm = path.dirname(__filename_esm);

const OUTPUT_DIR = path.resolve(__dirname_esm, '../../packages/core/src/generated');
const INTERNAL_DIR = path.join(OUTPUT_DIR, 'internal');

// ============================================================================
// Main
// ============================================================================

async function main() {
    console.log('🔧 v5 CodeGen: Generating operator code from OpRegistry...\n');

    // Ensure output directories exist
    if (!fs.existsSync(OUTPUT_DIR)) {
        fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    }
    if (!fs.existsSync(INTERNAL_DIR)) {
        fs.mkdirSync(INTERNAL_DIR, { recursive: true });
    }

    // Clear internal directory (except stub-utils.ts which is hand-written)
    const existingFiles = fs.readdirSync(INTERNAL_DIR);
    const filesToDelete = existingFiles.filter(f => f !== 'stub-utils.ts');
    for (const file of filesToDelete) {
        fs.unlinkSync(path.join(INTERNAL_DIR, file));
    }
    console.log(`🗑️  Cleared ${filesToDelete.length} files from internal/\n`);

    // Group entries by name
    const groupedEntries = new Map<string, typeof OpRegistry[number][]>();
    for (const entry of OpRegistry) {
        if (!groupedEntries.has(entry.name)) {
            groupedEntries.set(entry.name, []);
        }
        groupedEntries.get(entry.name)!.push(entry);
    }

    console.log(`📦 Found ${OpRegistry.length} OpEntries across ${groupedEntries.size} operators\n`);

    // Generate internal/*.ts (stub implementations)
    console.log('📝 Generating internal/*.ts (stub implementations)...');
    const internalStats = generateInternalFiles(OpRegistry, INTERNAL_DIR);
    console.log(`   ✅ Generated ${internalStats.count} internal files\n`);

    // Generate internal/index.ts
    const internalIndexCode = generateInternalIndex(OpRegistry);
    fs.writeFileSync(path.join(INTERNAL_DIR, 'index.ts'), internalIndexCode);
    console.log('   ✅ Generated internal/index.ts\n');

    // Generate ops.ts
    console.log('📝 Generating ops.ts...');
    const opsCode = generateOps(groupedEntries);
    fs.writeFileSync(path.join(OUTPUT_DIR, 'ops.ts'), opsCode);
    console.log('   ✅ Generated ops.ts\n');

    // Generate tensor-methods.ts
    console.log('📝 Generating tensor-methods.ts...');
    const methodsCode = generateTensorMethods(groupedEntries);
    fs.writeFileSync(path.join(OUTPUT_DIR, 'tensor-methods.ts'), methodsCode);
    console.log('   ✅ Generated tensor-methods.ts\n');

    // Generate types.d.ts
    console.log('📝 Generating types.d.ts...');
    const typesCode = generateTypes(groupedEntries);
    fs.writeFileSync(path.join(OUTPUT_DIR, 'types.d.ts'), typesCode);
    console.log('   ✅ Generated types.d.ts\n');

    // Generate index.ts
    const indexCode = `/**
 * v5 Generated Code
 * DO NOT EDIT - Generated from OpRegistry
 * Generated at: ${new Date().toISOString()}
 */
export * from './ops';
export * from './tensor-methods';
`;
    fs.writeFileSync(path.join(OUTPUT_DIR, 'index.ts'), indexCode);
    console.log('   ✅ Generated index.ts\n');

    console.log('✨ v5 CodeGen complete!\n');
    console.log(`Output directory: ${OUTPUT_DIR}`);
    console.log(`Total OpEntries: ${OpRegistry.length}`);
    console.log(`Total operators: ${groupedEntries.size}`);
}

/**
 * Generate internal/index.ts that exports all internal functions
 */
/**
 * Generate internal/index.ts that exports all internal functions
 */
function generateInternalIndex(entries: typeof OpRegistry): string {
    const lines: string[] = [];

    lines.push('/**');
    lines.push(' * v5 Internal Functions Index');
    lines.push(' * DO NOT EDIT - Generated from OpRegistry');
    lines.push(` * Generated at: ${new Date().toISOString()}`);
    lines.push(' */');
    lines.push('');

    // Group by mechanism for organized exports
    const byMechanism = new Map<string, string[]>();
    for (const entry of entries) {
        if (!byMechanism.has(entry.mechanism)) {
            byMechanism.set(entry.mechanism, []);
        }
        const funcName = entry.variant ? `${entry.name}_${entry.variant}` : entry.name;
        byMechanism.get(entry.mechanism)!.push(funcName);
    }

    // Sort mechanisms for stable output
    const sortedMechanisms = [...byMechanism.keys()].sort();

    for (const mech of sortedMechanisms) {
        const funcs = byMechanism.get(mech)!;
        lines.push(`// ${mech}`);
        for (const func of funcs) {
            lines.push(`export { ${func} } from './${func}';`);
        }
        lines.push('');
    }

    return lines.join('\n');
}

main().catch(console.error);
