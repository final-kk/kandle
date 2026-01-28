/**
 * Reduction Kernel Types
 * 
 * 定义归约操作的配置接口，遵循 PyTorch 的 TensorIterator 架构
 */

/**
 * 归约操作配置
 * 每个 dispatchKey 对应一个配置
 */
export interface ReductionOpConfig {
    /**
     * 初始值生成器
     * @param computeType - WGSL 类型 'f32' | 'i32' | 'u32'
     * @returns WGSL 表达式，如 "0.0" 或 "f32(-1e38)"
     */
    initializer: (computeType: string) => string;

    /**
     * 累加器 - 将新值合并到累计值
     * @param accumVar - 累积变量名，如 "acc"
     * @param valueVar - 新值变量名，如 "val"
     * @param computeType - WGSL 类型
     * @returns WGSL 赋值语句，如 "acc = acc + val;"
     */
    accumulator: (accumVar: string, valueVar: string, computeType: string) => string;

    /**
     * 终结器 - 将累计值转换为最终结果 (可选)
     * @param accumVar - 累积变量名
     * @param totalNumel - 总元素数变量名，如 "uniforms.totalReductionNumel"
     * @param computeType - WGSL 类型
     * @returns WGSL 表达式，如 "acc / f32(totalNumel)" (mean)
     */
    finalizer?: (accumVar: string, totalNumel: string, computeType: string) => string;

    /**
     * Scalar 默认值 (可选)
     * 例如: sum 没有 scalar 参数，不需要此字段
     */
    scalarDefaults?: Record<string, number>;
}
