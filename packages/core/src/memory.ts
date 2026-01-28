/**
 * Scope-based Tensor Memory Management
 * 
 * 提供类似 TensorFlow.js tidy() 的自动内存管理，
 * 自动释放 scope 内创建的所有中间张量。
 * 
 * 对标 TensorFlow.js engine.ts 的 tidy/startScope/endScope 实现。
 * 
 * @module memory
 */

import { Tensor } from './tensor';

// ============================================================================
// Types
// ============================================================================

/**
 * Scope 状态
 */
interface ScopeState {
    /** 在此 scope 中创建的所有张量 */
    track: Tensor[];
    /** Scope 唯一 ID */
    id: number;
    /** Scope 名称 (用于调试) */
    name: string;
}

// ============================================================================
// Global State
// ============================================================================

/** 下一个 scope ID */
let nextScopeId = 0;

/** 当前活跃的 scope (栈顶) */
let activeScope: ScopeState | null = null;

/** Scope 栈 */
const scopeStack: ScopeState[] = [];

// ============================================================================
// Core API
// ============================================================================

/**
 * 自动管理中间张量的生命周期 (同步版本)
 * 
 * 在回调函数执行期间创建的所有张量会被自动追踪。
 * 回调结束后，除了返回值外的所有张量都会被自动释放。
 * 
 * 对标 TensorFlow.js 的 tf.tidy() API。
 * 
 * **注意：不支持异步函数！** 如果需要在异步流程中使用，
 * 请在异步函数内部嵌套同步 tidy() 调用。
 * 
 * @param fn 要执行的同步函数
 * @returns fn 的返回值，其中的张量会被保留（不释放）
 * 
 * @example
 * ```ts
 * // 正确用法：同步函数
 * const result = tidy(() => {
 *     const a = randn([100, 100]);
 *     const b = randn([100, 100]);
 *     const c = matmul(a, b); // a, b 会在 tidy 结束后被释放
 *     return relu(c);         // 返回值会被保留
 * });
 * 
 * // 在异步流程中使用
 * async function inference() {
 *     const intermediate = tidy(() => {
 *         // 同步计算块
 *         return matmul(a, b);
 *     });
 *     const result = await someAsyncOp(intermediate);
 *     intermediate.dispose(); // 手动释放
 *     return result;
 * }
 * ```
 */
export function tidy<T>(fn: () => T): T;
export function tidy<T>(name: string, fn: () => T): T;
export function tidy<T>(nameOrFn: string | (() => T), fn?: () => T): T {
    let name: string = 'unnamed scope';
    let func: () => T;

    if (typeof nameOrFn === 'function') {
        func = nameOrFn;
    } else {
        name = nameOrFn;
        if (fn === undefined) {
            throw new Error('When calling tidy with a name, the second argument must be a function');
        }
        func = fn;
    }

    // 开始新的 scope
    startScope(name);

    let result: T;
    try {
        result = func();

        // 检测 Promise（不支持异步）
        if (result instanceof Promise) {
            console.error(
                'Cannot return a Promise inside of tidy. ' +
                'Use synchronous tidy() inside async functions instead.'
            );
        }
    } finally {
        // 结束 scope，释放未被返回的张量
        endScope(result!);
    }

    return result;
}

/**
 * 自动管理中间张量的生命周期 (异步版本)
 * 
 * 在回调函数执行期间创建的所有张量会被自动追踪。
 * 回调结束后，除了返回值外的所有张量都会被自动释放。
 * 
 * 这是专为 async/await 设计的版本，适用于模型推理等异步场景。
 * 
 * @param fn 要执行的异步函数
 * @returns fn 的返回值，其中的张量会被保留（不释放）
 * 
 * @example
 * ```ts
 * // 模型推理
 * const output = await tidyAsync(async () => {
 *     const hidden = await model.forward(input);
 *     const logits = await lmHead.forward(hidden);
 *     return logits;  // hidden 会被自动释放，logits 会被保留
 * });
 * 
 * // 使用 keep() 保留 KV Cache
 * await tidyAsync(async () => {
 *     const [k, v] = await computeKV(x);
 *     keep(k);  // k 不会被释放
 *     keep(v);  // v 不会被释放
 *     kvCache.update(layer, k, v);
 * });
 * ```
 */
export async function tidyAsync<T>(fn: () => Promise<T>): Promise<T>;
export async function tidyAsync<T>(name: string, fn: () => Promise<T>): Promise<T>;
export async function tidyAsync<T>(
    nameOrFn: string | (() => Promise<T>),
    fn?: () => Promise<T>
): Promise<T> {
    let name: string = 'unnamed async scope';
    let func: () => Promise<T>;

    if (typeof nameOrFn === 'function') {
        func = nameOrFn;
    } else {
        name = nameOrFn;
        if (fn === undefined) {
            throw new Error('When calling tidyAsync with a name, the second argument must be a function');
        }
        func = fn;
    }

    // 开始新的 scope
    startScope(name);

    try {
        const result = await func();
        // 结束 scope，释放未被返回的张量
        endScope(result);
        return result;
    } catch (e) {
        // 异常时释放所有张量（不保留任何返回值）
        endScope(undefined);
        throw e;
    }
}

/**
 * 标记一个张量为"保留"，使其不会被当前 tidy scope 自动释放
 * 
 * @param tensor 要保留的张量
 * @returns 相同的张量（方便链式调用）
 */
export function keep<T extends Tensor>(tensor: T): T {
    (tensor as any).kept = true;
    return tensor;
}

// ============================================================================
// Internal API
// ============================================================================

/**
 * 开始新的 scope
 * 
 * @internal
 */
function startScope(name: string = 'unnamed scope'): void {
    const scopeState: ScopeState = {
        track: [],
        id: nextScopeId++,
        name,
    };
    scopeStack.push(scopeState);
    activeScope = scopeState;
}

/**
 * 结束当前 scope，释放未被返回的张量
 * 
 * @internal
 */
function endScope(result: unknown): void {
    if (scopeStack.length === 0) {
        throw new Error('No active scope to end');
    }

    // 提取返回值中的所有张量
    const tensorsToKeep = extractTensors(result);
    const keepIds = new Set([...tensorsToKeep].map(t => t.id));

    // 弹出当前 scope
    const oldScope = scopeStack.pop()!;
    activeScope = scopeStack.length > 0
        ? scopeStack[scopeStack.length - 1]
        : null;

    // 释放未被保留的张量
    for (const tensor of oldScope.track) {
        const isKept = (tensor as any).kept === true;
        const isReturned = keepIds.has(tensor.id);
        const isDisposed = tensor.isDisposed;

        if (!isKept && !isReturned && !isDisposed) {
            tensor.dispose();
        }
    }

    // 将返回值中的张量转移到父 scope（如果有）
    if (activeScope != null) {
        for (const tensor of tensorsToKeep) {
            const isKept = (tensor as any).kept === true;
            // 只有非 kept 且是在 oldScope 创建的张量才需要转移到父 scope
            if (!isKept && (tensor as any).scopeId === oldScope.id) {
                activeScope.track.push(tensor);
                (tensor as any).scopeId = activeScope.id;
            }
        }
    }
}

/**
 * 从返回值中提取所有 Tensor，用于确定哪些需要保留
 */
function extractTensors(value: unknown): Set<Tensor> {
    const result = new Set<Tensor>();

    if (value instanceof Tensor) {
        result.add(value);
    } else if (Array.isArray(value)) {
        for (const item of value) {
            for (const t of extractTensors(item)) {
                result.add(t);
            }
        }
    } else if (value && typeof value === 'object') {
        for (const key of Object.keys(value)) {
            for (const t of extractTensors((value as any)[key])) {
                result.add(t);
            }
        }
    }

    return result;
}

/**
 * 内部追踪函数 - 供 Tensor 构造函数使用
 * 
 * 在活跃的 scope 中追踪新创建的张量。
 * 
 * @internal
 */
export function _trackTensor(tensor: unknown): void {
    if (activeScope != null && tensor instanceof Tensor) {
        (tensor as any).scopeId = activeScope.id;
        activeScope.track.push(tensor);
    }
}

/**
 * 用于在 Tensor 构造函数中追踪新创建的张量
 * 
 * @deprecated 请使用 _trackTensor
 */
export function trackTensor(tensor: Tensor): void {
    _trackTensor(tensor);
}

/**
 * 检查是否在活跃的 tidy scope 中
 */
export function isInTidyScope(): boolean {
    return activeScope != null;
}
