/**
 * Sequential - 顺序容器
 *
 * 按顺序执行多个模块，前一个模块的输出作为后一个模块的输入
 *
 * @see https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
 */

import { Tensor } from '../../tensor';
import { Module } from '../module';

/**
 * Sequential 容器
 *
 * @example
 * ```ts
 * const model = new Sequential(
 *   new nn.Linear(784, 256),
 *   new nn.ReLU(),
 *   new nn.Linear(256, 10),
 * );
 *
 * const output = model.call(input);
 * ```
 */
export class Sequential extends Module {
    /** 有序模块列表 */
    private _moduleList: Module[] = [];

    /**
     * 创建 Sequential 容器
     *
     * @param modules 按顺序执行的模块列表
     */
    constructor(...modules: Module[]) {
        super();
        for (let i = 0; i < modules.length; i++) {
            this.addModule(String(i), modules[i]);
            this._moduleList.push(modules[i]);
        }
    }

    /**
     * 按顺序执行所有模块
     *
     * @param input 输入张量
     * @returns 最终输出张量
     */
    async forward(input: Tensor): Promise<Tensor> {
        let x = input;
        for (const module of this._moduleList) {
            x = await module.call(x) as Tensor;
        }
        return x;
    }

    /**
     * 获取模块数量
     */
    get length(): number {
        return this._moduleList.length;
    }

    /**
     * 通过索引获取模块
     */
    get(index: number): Module {
        if (index < 0) {
            index = this._moduleList.length + index;
        }
        if (index < 0 || index >= this._moduleList.length) {
            throw new Error(`Index ${index} out of range for Sequential with ${this._moduleList.length} modules`);
        }
        return this._moduleList[index];
    }

    /**
     * 添加模块到末尾
     */
    append(module: Module): this {
        const idx = this._moduleList.length;
        this.addModule(String(idx), module);
        this._moduleList.push(module);
        return this;
    }

    /**
     * 可迭代
     */
    [Symbol.iterator](): Iterator<Module> {
        return this._moduleList[Symbol.iterator]();
    }
}
