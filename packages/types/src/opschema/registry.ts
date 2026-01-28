/**
 * NN-Kit Operator Schema v5 - Registry
 *
 * OpEntry 注册表 - 每个变体独立注册
 *
 * @module v5/registry
 */

import type { OpEntry, OpMechanism } from './types';
// v7 重组后的模块导入
import * as pointwise from './ops/pointwise';       // 合并 unary + arithmetic + comparison
import * as activation from './ops/activation';     // 激活函数
import * as reduction from './ops/reduction';       // 归约运算
import * as linalg from './ops/linalg';             // 线性代数
import * as triangular from './ops/triangular';     // 三角矩阵操作
import * as norm from './ops/norm';                 // 归一化
import * as shape from './ops/shape';               // 形状操作 (原 view)
import * as creation from './ops/creation';         // 创建操作 (原 factory)
import * as memory from './ops/memory';             // 内存操作 (原 copy)
import * as scan from './ops/scan';                 // 扫描运算
import * as sort from './ops/sort';                 // 排序操作
import * as conv from './ops/conv';                 // 卷积与池化
import * as indexing from './ops/indexing';         // 索引操作 (合并 gather + scatter)
import * as attention from './ops/attention';       // 注意力机制
import * as fft from './ops/fft';                   // FFT 变换

// ============================================================================
// 收集所有 OpEntry
// ============================================================================

function collectEntries(module: Record<string, unknown>): OpEntry[] {
    return Object.values(module).filter(
        (v): v is OpEntry => typeof v === 'object' && v !== null && 'name' in v && 'mechanism' in v
    );
}

/**
 * 所有 OpEntry 的扁平数组
 *
 * v7 核心变化: 按语义领域组织
 * 每个变体是独立的 entry（如 add_Tensor 和 add_Scalar 是两个独立的 OpEntry）
 */
export const OpRegistry: readonly OpEntry[] = [
    ...collectEntries(pointwise),     // 逐元素运算
    ...collectEntries(activation),    // 激活函数
    ...collectEntries(reduction),     // 归约运算
    ...collectEntries(linalg),        // 线性代数
    ...collectEntries(triangular),    // 三角矩阵
    ...collectEntries(norm),          // 归一化
    ...collectEntries(shape),         // 形状操作
    ...collectEntries(creation),      // 创建操作
    ...collectEntries(memory),        // 内存操作
    ...collectEntries(scan),          // 扫描运算
    ...collectEntries(sort),          // 排序操作
    ...collectEntries(conv),          // 卷积与池化
    ...collectEntries(indexing),      // 索引操作
    ...collectEntries(attention),     // 注意力机制
    ...collectEntries(fft),           // FFT 变换
];

// ============================================================================
// 索引和查询
// ============================================================================

/**
 * 按 (name, variant) 索引的 Map
 * key 格式: "add.Tensor" 或 "add" (无变体)
 */
const entryByKey = new Map<string, OpEntry>();

/**
 * 按 name 分组的 Map
 * 用于查找同一操作的所有变体
 */
const entriesByName = new Map<string, OpEntry[]>();

// 构建索引
for (const entry of OpRegistry) {
    const key = entry.variant ? `${entry.name}.${entry.variant}` : entry.name;
    entryByKey.set(key, entry);

    if (!entriesByName.has(entry.name)) {
        entriesByName.set(entry.name, []);
    }
    entriesByName.get(entry.name)!.push(entry);
}

/**
 * 获取特定 OpEntry
 * @param name 操作符名称
 * @param variant 变体名称 (可选)
 */
export function getOpEntry(name: string, variant?: string): OpEntry | undefined {
    const key = variant ? `${name}.${variant}` : name;
    return entryByKey.get(key);
}

/**
 * 获取操作符的所有变体
 */
export function getOpVariants(name: string): OpEntry[] {
    return entriesByName.get(name) ?? [];
}

/**
 * 按 Mechanism 筛选
 */
export function getOpsByMechanism(mechanism: OpMechanism): OpEntry[] {
    return OpRegistry.filter(entry => entry.mechanism === mechanism);
}

/**
 * 获取所有唯一的操作符名称
 */
export function getAllOpNames(): string[] {
    return [...entriesByName.keys()];
}

/**
 * 检查是否有多个变体
 */
export function hasVariants(name: string): boolean {
    const variants = entriesByName.get(name);
    return variants !== undefined && variants.length > 1;
}

// ============================================================================
// 类型辅助
// ============================================================================

/**
 * 所有操作符名称的联合类型
 */
export type OpName = typeof OpRegistry[number]['name'];

/**
 * 按 Mechanism 分组的操作符名称
 */
export const MechanismGroups = {
    Iterator: getOpsByMechanism('Iterator').map(e => e.name),
    Composite: getOpsByMechanism('Composite').map(e => e.name),
    View: getOpsByMechanism('View').map(e => e.name),
    Copy: getOpsByMechanism('Copy').map(e => e.name),
    Factory: getOpsByMechanism('Factory').map(e => e.name),
    // 专用 Kernel 机制
    Matrix: getOpsByMechanism('Matrix').map(e => e.name),
    Window: getOpsByMechanism('Window').map(e => e.name),
    Normalize: getOpsByMechanism('Normalize').map(e => e.name),
    Sort: getOpsByMechanism('Sort').map(e => e.name),
    Gather: getOpsByMechanism('Gather').map(e => e.name),
    Scatter: getOpsByMechanism('Scatter').map(e => e.name),
    Triangular: getOpsByMechanism('Triangular').map(e => e.name),
    Shape: getOpsByMechanism('Shape').map(e => e.name),
} as const;
