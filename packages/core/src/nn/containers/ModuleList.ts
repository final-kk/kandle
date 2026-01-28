/**
 * ModuleList - 模块列表容器
 *
 * 用于保存多个子模块的列表，支持迭代但没有 forward 方法
 *
 * @see https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
 */

import { Tensor } from '../../tensor';
import { Module } from '../module';

/**
 * ModuleList 容器
 *
 * 用于需要动态层数的场景，如 Transformer layers
 *
 * @example
 * ```ts
 * class LLaMA extends Module {
 *   layers: ModuleList;
 *
 *   constructor(config: Config) {
 *     super();
 *     this.layers = new ModuleList(
 *       Array.from({ length: config.numLayers }, () =>
 *         new TransformerBlock(config)
 *       )
 *     );
 *   }
 *
 *   forward(x: Tensor): Tensor {
 *     for (const layer of this.layers) {
 *       x = layer.call(x) as Tensor;
 *     }
 *     return x;
 *   }
 * }
 * ```
 */
export class ModuleList extends Module implements Iterable<Module> {
    /** 内部模块列表 */
    private _list: Module[] = [];

    /**
     * 创建 ModuleList
     *
     * @param modules 可选的初始模块列表
     */
    constructor(modules?: Module[]) {
        super();
        if (modules) {
            for (const module of modules) {
                this.append(module);
            }
        }
    }

    /**
     * 添加模块到末尾
     */
    append(module: Module): this {
        const idx = this._list.length;
        this.addModule(String(idx), module);
        this._list.push(module);
        return this;
    }

    /**
     * 在指定位置插入模块
     */
    insert(index: number, module: Module): this {
        if (index < 0 || index > this._list.length) {
            throw new Error(`Index ${index} out of range for insert`);
        }

        // 插入到列表
        this._list.splice(index, 0, module);

        // 重新注册所有模块 (因为索引变了)
        this._modules.clear();
        for (let i = 0; i < this._list.length; i++) {
            this.addModule(String(i), this._list[i]);
        }

        return this;
    }

    /**
     * 扩展列表
     */
    extend(modules: Module[]): this {
        for (const module of modules) {
            this.append(module);
        }
        return this;
    }

    /**
     * 获取模块数量
     */
    get length(): number {
        return this._list.length;
    }

    /**
     * 通过索引获取模块
     */
    get(index: number): Module {
        if (index < 0) {
            index = this._list.length + index;
        }
        if (index < 0 || index >= this._list.length) {
            throw new Error(`Index ${index} out of range for ModuleList with ${this._list.length} modules`);
        }
        return this._list[index];
    }

    /**
     * 通过索引设置模块
     */
    set(index: number, module: Module): void {
        if (index < 0) {
            index = this._list.length + index;
        }
        if (index < 0 || index >= this._list.length) {
            throw new Error(`Index ${index} out of range for ModuleList with ${this._list.length} modules`);
        }
        this._list[index] = module;
        this.addModule(String(index), module);
    }

    /**
     * 可迭代
     */
    [Symbol.iterator](): Iterator<Module> {
        return this._list[Symbol.iterator]();
    }

    /**
     * ModuleList 没有 forward 方法
     *
     * 用户应该在自己的模块中迭代 ModuleList
     */
    forward(_input: Tensor): Promise<Tensor> {
        throw new Error('ModuleList has no forward method. Iterate over it in your own forward method.');
    }

    /**
     * 额外表示信息
     */
    protected extraRepr(): string {
        return '';
    }
}
