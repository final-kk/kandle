/**
 * Module - 神经网络模块基类
 *
 * 对标 PyTorch nn.Module，提供:
 * - 参数和缓冲区管理
 * - 子模块管理
 * - 状态字典序列化/反序列化
 * - 训练/推理模式切换
 * - Hook 机制 (预留)
 *
 * @see https://pytorch.org/docs/stable/generated/torch.nn.Module.html
 */

import type { DType, DeviceNameEnum, Shape } from '@kandle/types';
import { Tensor } from '../tensor';
import { Parameter, isParameter } from './parameter';
import type { SafetensorGroup } from '../io/safetensor/types';
import { tensorFromSafetensorLayer } from '../io/tensor-loaders';

// ============================================================================
// Types
// ============================================================================

/**
 * Forward pre-hook 签名
 * 在 forward 之前调用，可以修改输入
 */
export type ForwardPreHook = (module: Module, input: any[]) => any[] | void;

/**
 * Forward hook 签名
 * 在 forward 之后调用，可以修改输出
 */
export type ForwardHook = (module: Module, input: any[], output: any) => any | void;

/**
 * 可移除的 hook 句柄
 */
export interface RemovableHandle {
    remove(): void;
}

/**
 * loadStateDict 选项
 */
export interface LoadStateDictOptions {
    /**
     * 是否严格匹配键
     * - true: 所有键必须匹配，否则抛出错误
     * - false: 允许缺失和多余的键
     * @default true
     */
    strict?: boolean;

    /**
     * 赋值模式
     * - 'copy': 拷贝数据 (当前未实现 copy_，暂时与 reference 相同)
     * - 'reference': 直接共享 handle (零拷贝)
     * @default 'reference'
     */
    assignMode?: 'copy' | 'reference';
}

/**
 * loadStateDict 结果
 */
export interface LoadStateDictResult {
    /** 模型期望但 stateDict 中缺失的键 */
    missingKeys: string[];
    /** stateDict 中存在但模型不需要的键 */
    unexpectedKeys: string[];
    /** 成功加载的键 */
    loadedKeys: string[];
}

/**
 * loadFromSafetensor 选项
 */
export interface LoadFromSafetensorOptions {
    /** 是否严格匹配键 */
    strict?: boolean;
    /** 目标设备 */
    device?: DeviceNameEnum;
    /** 目标 dtype */
    dtype?: DType;
    /** 键名映射函数 */
    keyMapper?: (key: string) => string;
    /** 取消信号 */
    signal?: AbortSignal;
    /** 
     * 并发加载批次大小 (默认 4)
     * 控制同时加载的 tensor 数量，避免浏览器并发限制
     */
    batchSize?: number;
}

// ============================================================================
// Module Base Class
// ============================================================================

/**
 * Module 基类
 *
 * 所有神经网络层/模块都应继承此类
 */
export abstract class Module {
    // ========================================================================
    // Internal State
    // ========================================================================

    /** 可学习参数 Map */
    protected _parameters: Map<string, Parameter | null> = new Map();

    /** 持久化缓冲区 (如 BatchNorm 的 running_mean) */
    protected _buffers: Map<string, Tensor | null> = new Map();

    /** 子模块 Map */
    protected _modules: Map<string, Module | null> = new Map();

    /** 训练模式标志 */
    protected _training: boolean = true;

    /** 非持久化缓冲区名称集合 (不保存到 state_dict) */
    protected _nonPersistentBuffersSet: Set<string> = new Set();

    // Hooks (预留)
    private _forwardPreHooks: Map<number, ForwardPreHook> = new Map();
    private _forwardHooks: Map<number, ForwardHook> = new Map();
    private _hookIdCounter: number = 0;

    // ========================================================================
    // Constructor
    // ========================================================================

    constructor() {
        // 基类构造函数无需特殊处理
    }

    // ========================================================================
    // Abstract Methods
    // ========================================================================

    /**
     * 前向传播 - 子类必须实现
     *
     * 注意: 不要直接调用此方法，应该使用 module.call() 来触发 hooks
     *
     * 设计说明:
     * 由于 WebGPU 数据访问是异步的 (GPUBuffer.mapAsync())，用户在自定义 Module
     * 的 forward 中可能需要异步访问 tensor 数据（如 debug 场景）。
     *
     * 由于 async 传染性，一旦调用链中有任一环节是异步的，整个链路必须异步。
     * 因此统一使用 Promise 返回类型。
     *
     * 对于纯 GPU 操作的内置 Module，虽然签名是 async，但内部没有 await，
     * 性能开销可忽略（V8 对此高度优化）。
     */
    abstract forward(...args: any[]): Promise<Tensor | Tensor[] | any>;

    // ========================================================================
    // Call (Entry Point)
    // ========================================================================

    /**
     * 调用模块的入口点
     *
     * 正确触发 pre-hooks 和 post-hooks
     *
     * @example
     * ```ts
     * const output = await model.call(input);
     * // 或使用 callable getter
     * const fn = model.callable;
     * const output = await fn(input);
     * ```
     */
    async call(...args: any[]): Promise<Tensor | Tensor[] | any> {
        // 1. Pre-hooks (支持同步和异步)
        for (const hook of this._forwardPreHooks.values()) {
            const result = hook(this, args);
            if (result !== undefined) {
                // 支持异步 hook
                args = result instanceof Promise ? await result : result;
            }
        }

        // 2. Forward (异步)
        let output = await this.forward(...args);

        // 3. Post-hooks (支持同步和异步)
        for (const hook of this._forwardHooks.values()) {
            const hookResult = hook(this, args, output);
            if (hookResult !== undefined) {
                // 支持异步 hook
                output = hookResult instanceof Promise ? await hookResult : hookResult;
            }
        }

        return output;
    }

    /**
     * 获取可调用的异步函数形式
     *
     * 允许像函数一样使用模块:
     * ```ts
     * const fn = model.callable;
     * const output = await fn(input);
     * ```
     *
     * 注意: JavaScript 不支持 __call__ 魔法方法
     */
    get callable(): (...args: any[]) => Promise<Tensor | Tensor[] | any> {
        return this.call.bind(this);
    }

    // ========================================================================
    // Registration Methods
    // ========================================================================

    /**
     * 注册一个参数
     *
     * @param name 参数名称 (不能包含 '.')
     * @param param 参数对象，或 null 表示可选参数未使用
     */
    registerParameter(name: string, param: Parameter | null): void {
        if (name.includes('.')) {
            throw new Error(`Parameter name cannot contain '.': ${name}`);
        }
        if (name === '') {
            throw new Error('Parameter name cannot be empty string');
        }
        this._parameters.set(name, param);
    }

    /**
     * 注册一个缓冲区
     *
     * 缓冲区是模块状态的一部分，但不是可学习参数 (如 BatchNorm 的 running_mean)
     *
     * @param name 缓冲区名称 (不能包含 '.')
     * @param tensor 缓冲区张量
     * @param persistent 是否持久化 (保存到 state_dict)
     */
    registerBuffer(name: string, tensor: Tensor | null, persistent: boolean = true): void {
        if (name.includes('.')) {
            throw new Error(`Buffer name cannot contain '.': ${name}`);
        }
        if (name === '') {
            throw new Error('Buffer name cannot be empty string');
        }
        this._buffers.set(name, tensor);
        if (!persistent) {
            this._nonPersistentBuffersSet.add(name);
        } else {
            this._nonPersistentBuffersSet.delete(name);
        }
    }

    /**
     * 添加子模块
     *
     * @param name 模块名称 (不能包含 '.')
     * @param module 子模块对象
     */
    addModule(name: string, module: Module | null): void {
        if (name.includes('.')) {
            throw new Error(`Module name cannot contain '.': ${name}`);
        }
        if (name === '') {
            throw new Error('Module name cannot be empty string');
        }
        this._modules.set(name, module);
    }

    /**
     * 自动扫描并注册子模块和参数
     *
     * 遍历 this 上的所有属性，自动注册:
     * - Module 实例 → _modules
     * - Parameter 实例 → _parameters
     *
     * 用户应在构造函数末尾调用此方法
     *
     * @example
     * ```ts
     * class MyModel extends Module {
     *   linear1: nn.Linear;
     *   linear2: nn.Linear;
     *
     *   constructor() {
     *     super();
     *     this.linear1 = new nn.Linear(10, 20);
     *     this.linear2 = new nn.Linear(20, 5);
     *     this._registerChildren(); // 自动注册
     *   }
     * }
     * ```
     */
    protected _registerChildren(): void {
        for (const key of Object.getOwnPropertyNames(this)) {
            // 跳过私有属性和已注册的
            if (key.startsWith('_')) continue;

            const value = (this as any)[key];

            if (value instanceof Module && !this._modules.has(key)) {
                this._modules.set(key, value);
            } else if (isParameter(value) && !this._parameters.has(key)) {
                this._parameters.set(key, value);
            }
        }
    }

    // ========================================================================
    // Iterators
    // ========================================================================

    /**
     * 获取所有参数的迭代器
     *
     * @param recurse 是否递归包含子模块的参数
     */
    *parameters(recurse: boolean = true): IterableIterator<Parameter> {
        for (const param of this._parameters.values()) {
            if (param !== null) {
                yield param;
            }
        }
        if (recurse) {
            for (const module of this._modules.values()) {
                if (module !== null) {
                    yield* module.parameters(true);
                }
            }
        }
    }

    /**
     * 获取所有命名参数的迭代器
     *
     * @param prefix 键名前缀
     * @param recurse 是否递归包含子模块
     */
    *namedParameters(prefix: string = '', recurse: boolean = true): IterableIterator<[string, Parameter]> {
        const genPrefix = prefix ? prefix + '.' : '';
        for (const [name, param] of this._parameters) {
            if (param !== null) {
                yield [genPrefix + name, param];
            }
        }
        if (recurse) {
            for (const [name, module] of this._modules) {
                if (module !== null) {
                    yield* module.namedParameters(genPrefix + name, true);
                }
            }
        }
    }

    /**
     * 获取所有缓冲区的迭代器
     *
     * @param recurse 是否递归包含子模块
     */
    *buffers(recurse: boolean = true): IterableIterator<Tensor> {
        for (const buffer of this._buffers.values()) {
            if (buffer !== null) {
                yield buffer;
            }
        }
        if (recurse) {
            for (const module of this._modules.values()) {
                if (module !== null) {
                    yield* module.buffers(true);
                }
            }
        }
    }

    /**
     * 获取所有命名缓冲区的迭代器
     *
     * @param prefix 键名前缀
     * @param recurse 是否递归包含子模块
     */
    *namedBuffers(prefix: string = '', recurse: boolean = true): IterableIterator<[string, Tensor]> {
        const genPrefix = prefix ? prefix + '.' : '';
        for (const [name, buffer] of this._buffers) {
            if (buffer !== null) {
                yield [genPrefix + name, buffer];
            }
        }
        if (recurse) {
            for (const [name, module] of this._modules) {
                if (module !== null) {
                    yield* module.namedBuffers(genPrefix + name, true);
                }
            }
        }
    }

    /**
     * 获取直接子模块的迭代器 (非递归)
     */
    *children(): IterableIterator<Module> {
        for (const module of this._modules.values()) {
            if (module !== null) {
                yield module;
            }
        }
    }

    /**
     * 获取命名子模块的迭代器 (非递归)
     */
    *namedChildren(): IterableIterator<[string, Module]> {
        for (const [name, module] of this._modules) {
            if (module !== null) {
                yield [name, module];
            }
        }
    }

    /**
     * 获取所有模块的迭代器 (包括 self，递归)
     */
    *modules(): IterableIterator<Module> {
        yield this;
        for (const module of this._modules.values()) {
            if (module !== null) {
                yield* module.modules();
            }
        }
    }

    /**
     * 获取所有命名模块的迭代器 (包括 self，递归)
     */
    *namedModules(prefix: string = ''): IterableIterator<[string, Module]> {
        yield [prefix, this];
        for (const [name, module] of this._modules) {
            if (module !== null) {
                const subPrefix = prefix ? prefix + '.' + name : name;
                yield* module.namedModules(subPrefix);
            }
        }
    }

    // ========================================================================
    // Serialization
    // ========================================================================

    /**
     * 获取状态字典
     *
     * 返回包含所有参数和持久化缓冲区的 Map
     *
     * @param prefix 键名前缀
     */
    stateDict(prefix: string = ''): Map<string, Tensor> {
        const result = new Map<string, Tensor>();

        // 添加本模块的参数
        for (const [name, param] of this._parameters) {
            if (param !== null) {
                result.set(prefix + name, param);
            }
        }

        // 添加持久化缓冲区
        for (const [name, buffer] of this._buffers) {
            if (buffer !== null && !this._nonPersistentBuffersSet.has(name)) {
                result.set(prefix + name, buffer);
            }
        }

        // 递归子模块
        for (const [name, module] of this._modules) {
            if (module !== null) {
                const subPrefix = prefix ? prefix + name + '.' : name + '.';
                const subState = module.stateDict(subPrefix);
                for (const [k, v] of subState) {
                    result.set(k, v);
                }
            }
        }

        return result;
    }

    /**
     * 加载状态字典
     *
     * 将 stateDict 中的权重加载到模块
     *
     * @param stateDict 状态字典
     * @param options 加载选项
     */
    loadStateDict(
        stateDict: Map<string, Tensor>,
        options: LoadStateDictOptions = {}
    ): LoadStateDictResult {
        const {
            strict = true,
            assignMode = 'reference',
        } = options;

        const missingKeys: string[] = [];
        const unexpectedKeys: string[] = [];
        const loadedKeys: string[] = [];

        // 获取模型期望的所有键
        const expectedKeys = new Set<string>();
        for (const [name] of this.namedParameters('', true)) {
            expectedKeys.add(name);
        }
        for (const [name] of this.namedBuffers('', true)) {
            expectedKeys.add(name);
        }

        // 处理每个 state_dict 键
        for (const [key, tensor] of stateDict) {
            if (!expectedKeys.has(key)) {
                unexpectedKeys.push(key);
                continue;
            }

            // 查找目标参数/缓冲区并加载
            const loaded = this._loadKeyToModule(key, tensor, assignMode);
            if (loaded) {
                loadedKeys.push(key);
            } else {
                // 这不应该发生，因为我们已经检查了 expectedKeys
                missingKeys.push(key);
            }
        }

        // 检查未加载的键
        for (const expected of expectedKeys) {
            if (!loadedKeys.includes(expected)) {
                missingKeys.push(expected);
            }
        }

        // strict 模式下报错
        if (strict && (missingKeys.length > 0 || unexpectedKeys.length > 0)) {
            throw new Error(
                `Error(s) in loading state_dict:\n` +
                `  Missing keys: [${missingKeys.join(', ')}]\n` +
                `  Unexpected keys: [${unexpectedKeys.join(', ')}]`
            );
        }

        return { missingKeys, unexpectedKeys, loadedKeys };
    }

    /**
     * 从 SafetensorGroup 加载权重
     * 
     * @param group - Safetensor 组
     * @param options - 加载选项
     * @returns 加载结果
     * 
     * @example
     * const model = new Qwen3Model(config);
     * const group = await loadSafetensor('./model.safetensors.index.json');
     * const result = await model.loadFromSafetensor(group, { strict: true });
     * console.log('Loaded:', result.loadedKeys.length);
     */
    async loadFromSafetensor(
        group: SafetensorGroup,
        options: LoadFromSafetensorOptions = {}
    ): Promise<LoadStateDictResult> {
        const {
            strict = true,
            device,
            dtype,
            keyMapper = (k) => k,
            signal,
            batchSize = 4,
        } = options;

        const loadedKeys: string[] = [];
        const missingKeys: string[] = [];
        const unexpectedKeys: string[] = [];

        // 1. 收集模型所有参数键
        const expectedKeys = new Set<string>();
        for (const [name] of this.namedParameters('', true)) {
            expectedKeys.add(name);
        }
        for (const [name] of this.namedBuffers('', true)) {
            expectedKeys.add(name);
        }

        // 2. 构建 safetensor key → model key 的映射
        const keyToModelKey = new Map<string, string>();
        for (const layerName of group.layers.keys()) {
            const modelKey = keyMapper(layerName);
            keyToModelKey.set(layerName, modelKey);

            if (!expectedKeys.has(modelKey)) {
                unexpectedKeys.push(layerName);
            }
        }

        // 3. 收集需要加载的任务
        interface LoadTask {
            expectedKey: string;
            layerName: string;
        }

        const tasks: LoadTask[] = [];
        for (const expectedKey of expectedKeys) {
            // 查找对应的 safetensor layer
            let foundLayerName: string | null = null;
            for (const [layerName, modelKey] of keyToModelKey) {
                if (modelKey === expectedKey) {
                    foundLayerName = layerName;
                    break;
                }
            }

            if (!foundLayerName) {
                missingKeys.push(expectedKey);
                continue;
            }

            tasks.push({ expectedKey, layerName: foundLayerName });
        }

        // 4. 分批加载 (控制并发)
        for (let i = 0; i < tasks.length; i += batchSize) {
            // 检查取消信号
            if (signal?.aborted) {
                throw new DOMException('Aborted', 'AbortError');
            }

            const batch = tasks.slice(i, i + batchSize);

            // 并发加载这一批
            const results = await Promise.all(
                batch.map(async (task) => {
                    const layer = group.getLayer(task.layerName)!;
                    const tensor = await tensorFromSafetensorLayer(layer, { device, dtype }, signal);
                    return { task, tensor };
                })
            );

            // 加载到模型
            for (const { task, tensor } of results) {
                this._loadKeyToModule(task.expectedKey, tensor, 'reference');
                loadedKeys.push(task.expectedKey);
            }
        }

        // 5. strict 模式检查
        if (strict && (missingKeys.length > 0 || unexpectedKeys.length > 0)) {
            throw new Error(
                `Error(s) in loading state_dict:\n` +
                `  Missing keys: [${missingKeys.join(', ')}]\n` +
                `  Unexpected keys: [${unexpectedKeys.join(', ')}]`
            );
        }

        return { loadedKeys, missingKeys, unexpectedKeys };
    }

    /**
     * 递归查找并加载单个键
     */
    private _loadKeyToModule(
        key: string,
        tensor: Tensor,
        assignMode: 'copy' | 'reference'
    ): boolean {
        const parts = key.split('.');

        if (parts.length === 1) {
            // 直接属于本模块
            const name = parts[0];
            if (this._parameters.has(name)) {
                const param = this._parameters.get(name);
                if (param !== null && param !== undefined) {
                    this._assignTensor(param, tensor, assignMode);
                    return true;
                }
            }
            if (this._buffers.has(name)) {
                const buffer = this._buffers.get(name);
                if (buffer !== null && buffer !== undefined) {
                    this._assignTensor(buffer, tensor, assignMode);
                    return true;
                }
            }
            return false;
        }

        // 递归到子模块
        const [moduleName, ...rest] = parts;
        const subModule = this._modules.get(moduleName);
        if (subModule) {
            return subModule._loadKeyToModule(rest.join('.'), tensor, assignMode);
        }

        return false;
    }

    /**
     * 赋值张量数据
     */
    private _assignTensor(
        target: Tensor | Parameter,
        source: Tensor,
        mode: 'copy' | 'reference'
    ): void {
        // 检查形状匹配
        if (!this._shapeEquals(target.shape, source.shape)) {
            throw new Error(
                `Shape mismatch: expected [${target.shape}], got [${source.shape}]`
            );
        }

        if (mode === 'reference') {
            // 直接替换 handle (零拷贝)
            (target as any)._handle = source._handle;
        } else {
            // copy 模式: 需要 copy_ 操作支持
            // TODO: 实现 copy_ 后改为真正的拷贝
            // 目前先用 reference 模式
            (target as any)._handle = source._handle;
        }
    }

    /**
     * 形状比较
     */
    private _shapeEquals(a: readonly number[], b: readonly number[]): boolean {
        if (a.length !== b.length) return false;
        for (let i = 0; i < a.length; i++) {
            if (a[i] !== b[i]) return false;
        }
        return true;
    }

    // ========================================================================
    // Mode Switching
    // ========================================================================

    /**
     * 获取当前训练模式
     */
    get training(): boolean {
        return this._training;
    }

    /**
     * 设置训练模式
     *
     * @param mode true=训练模式，false=评估模式
     * @returns this (链式调用)
     */
    train(mode: boolean = true): this {
        this._training = mode;
        for (const module of this._modules.values()) {
            if (module !== null) {
                module.train(mode);
            }
        }
        return this;
    }

    /**
     * 设置为评估模式
     *
     * 等价于 train(false)
     */
    eval(): this {
        return this.train(false);
    }

    // ========================================================================
    // Device/Dtype Conversion (预留)
    // ========================================================================

    /**
     * 转换设备或数据类型
     *
     * TODO: 实现真正的设备转换
     */
    to(deviceOrDtype: DeviceNameEnum | DType, dtype?: DType): this {
        // 遍历所有参数和缓冲区
        // 对每个调用 tensor.to()
        throw new Error('Module.to() is not implemented yet');
    }

    // ========================================================================
    // Hooks
    // ========================================================================

    /**
     * 注册 forward pre-hook
     *
     * Pre-hook 在 forward() 之前调用，可以修改输入
     */
    registerForwardPreHook(hook: ForwardPreHook): RemovableHandle {
        const id = this._hookIdCounter++;
        this._forwardPreHooks.set(id, hook);
        return {
            remove: () => {
                this._forwardPreHooks.delete(id);
            }
        };
    }

    /**
     * 注册 forward hook
     *
     * Hook 在 forward() 之后调用，可以修改输出
     */
    registerForwardHook(hook: ForwardHook): RemovableHandle {
        const id = this._hookIdCounter++;
        this._forwardHooks.set(id, hook);
        return {
            remove: () => {
                this._forwardHooks.delete(id);
            }
        };
    }

    // ========================================================================
    // String Representation
    // ========================================================================

    /**
     * 获取模块名称 (类名)
     */
    get name(): string {
        return this.constructor.name;
    }

    /**
     * 获取额外的表示信息 (子类可重写)
     */
    protected extraRepr(): string {
        return '';
    }

    /**
     * 字符串表示
     */
    toString(): string {
        const lines: string[] = [];
        const extra = this.extraRepr();

        if (this._modules.size === 0) {
            // 叶子模块
            return extra ? `${this.name}(${extra})` : `${this.name}()`;
        }

        // 有子模块
        lines.push(`${this.name}(`);
        for (const [name, module] of this._modules) {
            if (module !== null) {
                const moduleStr = module.toString().split('\n').map((line, i) =>
                    i === 0 ? line : '  ' + line
                ).join('\n');
                lines.push(`  (${name}): ${moduleStr}`);
            }
        }
        lines.push(')');
        return lines.join('\n');
    }

    [Symbol.for('nodejs.util.inspect.custom')](): string {
        return this.toString();
    }
}
