/**
 * AmplitudeToDB Transform
 *
 * 对标 torchaudio.transforms.AmplitudeToDB
 */

import { type Tensor } from "../../tensor";
import { Module } from "../../nn/module";
import { amplitudeToDB as amplitudeToDBFn } from "../functional";

export interface AmplitudeToDBOptions {
    /** 类型: 'power' 或 'magnitude' */
    stype?: "power" | "magnitude";
    /** 动态范围上限 (dB) */
    top_db?: number | null;
}

/**
 * AmplitudeToDB 变换类
 *
 * 将振幅/功率谱转换为分贝刻度
 *
 * @example
 * ```ts
 * const transform = new AmplitudeToDB({ stype: 'power', top_db: 80 });
 * const dbSpec = await transform.forward(powerSpec);
 * ```
 */
export class AmplitudeToDB extends Module {
    private multiplier: number;
    private amin: number;
    private top_db?: number;

    constructor(options: AmplitudeToDBOptions = {}) {
        super();

        const stype = options.stype ?? "power";

        // multiplier: 10 for power, 20 for magnitude
        this.multiplier = stype === "power" ? 10.0 : 20.0;
        this.amin = 1e-10;
        this.top_db = options.top_db ?? undefined;
    }

    /**
     * 将输入转换为分贝刻度
     *
     * @param x - 输入张量 (振幅或功率)
     * @returns 分贝刻度的张量
     */
    async forward(x: Tensor): Promise<Tensor> {
        return amplitudeToDBFn(x, this.multiplier, this.amin, 1.0, this.top_db);
    }
}
