/**
 * ModuleDict - 模块字典容器
 *
 * 用于保存命名子模块的字典
 *
 * @see https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html
 */

import { Tensor } from '../../tensor';
import { Module } from '../module';

/**
 * ModuleDict 容器
 *
 * @example
 * ```ts
 * const modules = new ModuleDict({
 *   encoder: new Encoder(),
 *   decoder: new Decoder(),
 * });
 *
 * const encoded = modules.get('encoder').call(input);
 * ```
 */
export class ModuleDict extends Module {
    /**
     * 创建 ModuleDict
     *
     * @param modules 可选的初始模块字典
     */
    constructor(modules?: Record<string, Module>) {
        super();
        if (modules) {
            for (const [name, module] of Object.entries(modules)) {
                this.addModule(name, module);
            }
        }
    }

    /**
     * 通过名称获取模块
     */
    get(name: string): Module | undefined {
        return this._modules.get(name) ?? undefined;
    }

    /**
     * 设置模块
     */
    set(name: string, module: Module): void {
        this.addModule(name, module);
    }

    /**
     * 删除模块
     */
    delete(name: string): boolean {
        return this._modules.delete(name);
    }

    /**
     * 检查是否包含模块
     */
    has(name: string): boolean {
        return this._modules.has(name);
    }

    /**
     * 获取所有键
     */
    keys(): IterableIterator<string> {
        return this._modules.keys();
    }

    /**
     * 获取所有值
     */
    values(): IterableIterator<Module> {
        return this.children();
    }

    /**
     * 获取所有键值对
     */
    entries(): IterableIterator<[string, Module]> {
        return this.namedChildren();
    }

    /**
     * 获取模块数量
     */
    get length(): number {
        return this._modules.size;
    }

    /**
     * 更新多个模块
     */
    update(modules: Record<string, Module>): void {
        for (const [name, module] of Object.entries(modules)) {
            this.addModule(name, module);
        }
    }

    /**
     * 清空所有模块
     */
    clear(): void {
        this._modules.clear();
    }

    /**
     * 弹出模块
     */
    pop(name: string): Module | undefined {
        const module = this._modules.get(name);
        if (module !== undefined) {
            this._modules.delete(name);
        }
        return module ?? undefined;
    }

    /**
     * 可迭代 (返回键值对)
     */
    [Symbol.iterator](): Iterator<[string, Module]> {
        return this._modules.entries() as Iterator<[string, Module]>;
    }

    /**
     * ModuleDict 没有 forward 方法
     *
     * 用户应该通过 get() 访问特定模块并调用其 forward
     */
    forward(_input: Tensor): Promise<Tensor> {
        throw new Error('ModuleDict has no forward method. Access modules by key and call their forward methods.');
    }
}
