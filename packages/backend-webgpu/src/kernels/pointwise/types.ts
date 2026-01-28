/**
 * Pointwise Kernel Types
 *
 * v5 架构: 表达式配置系统
 * 
 * Complex Number Support:
 * - Complex numbers are stored as vec2<f32> where x=real, y=imaginary
 * - Operations on complex types need specialized expressions
 * - When computeType is 'vec2<f32>', the complexExpr is used instead of expr
 */

/**
 * Pointwise 操作配置
 * 每个 dispatchKey 对应一个配置
 */
export interface PointwiseOpConfig {
    /**
     * WGSL 表达式生成器 (标量类型: f32, i32, u32)
     * @param inputs - 输入变量名 ['a'] 或 ['a', 'b'] 或 ['a', 'b', 'c']
     * @param scalars - Scalar 引用 { alpha: 'uniforms.alpha', ... }
     * @param computeType - WGSL 类型 'f32' | 'i32' | 'u32'
     */
    expr: (
        inputs: string[],
        scalars: Record<string, string>,
        computeType: string
    ) => string;

    /**
     * 复数类型的 WGSL 表达式生成器 (vec2<f32>)
     * 当 computeType === 'vec2<f32>' 时使用此表达式
     * 如果未定义但操作需要复数支持，将抛出运行时错误
     * 
     * 复数运算规则 (vec2<f32> = [real, imag]):
     * - add: (a.x + b.x, a.y + b.y) = a + b
     * - mul: (a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x)
     * - div: complex division formula
     * - abs: sqrt(a.x*a.x + a.y*a.y) -> returns f32 (modulus)
     * - neg: (-a.x, -a.y) = -a
     */
    complexExpr?: (
        inputs: string[],
        scalars: Record<string, string>
    ) => string;

    /**
     * 标记此操作是否支持复数输入
     * 如果为 false 或未定义，当遇到复数输入时会尝试使用 complexExpr
     * 如果 complexExpr 也未定义，将抛出错误
     */
    supportsComplex?: boolean;

    /**
     * Scalar 默认值
     * 当 iter.getScalarArg(name) 返回 undefined 时使用
     */
    scalarDefaults?: Record<string, number>;

    /**
     * Optional Scalar 的哨兵值
     * 当 Scalar 为 undefined 时，使用哨兵值代替
     * 例如: clamp 的 min=-1e38, max=1e38
     */
    scalarSentinels?: Record<string, number>;

    /**
     * WGSL 辅助函数
     * 当操作需要辅助函数时使用 (如 i0, sinc)
     * 这些函数会被添加到 shader 顶部
     */
    helperFunctions?: string[];
}

/**
 * 比较操作的输出类型标记
 */
export type OutputKind = 'same' | 'bool';

/**
 * 扩展的 Pointwise 操作配置 (包含输出类型信息)
 */
export interface ExtendedPointwiseOpConfig extends PointwiseOpConfig {
    /**
     * 输出类型
     * - 'same': 与输入相同类型
     * - 'bool': 输出布尔类型 (比较操作)
     */
    outputKind?: OutputKind;
}
