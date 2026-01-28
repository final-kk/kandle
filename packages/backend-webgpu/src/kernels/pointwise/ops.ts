/**
 * Pointwise Operations Registry
 *
 * v5 架构: 所有 Pointwise 操作的 WGSL 表达式配置
 *
 * 单一真相源: 添加新操作只需在此添加一行配置
 *
 * Complex Number Support:
 * - Complex numbers are stored as vec2<f32> where x=real, y=imaginary
 * - Operations with `complexExpr` support complex inputs
 * - Arithmetic: add, sub work naturally with vec2 (component-wise)
 * - mul/div need specialized formulas
 * - Some functions (abs, square) return real values from complex inputs
 */

import type { ExtendedPointwiseOpConfig } from './types';
import { WGSL_CONSTANTS } from '../../base/dtype';

// ============================================================================
// Complex Arithmetic Helper Expressions
// ============================================================================

/**
 * Complex multiplication: (a.x + a.y*i) * (b.x + b.y*i)
 * = (a.x*b.x - a.y*b.y) + (a.x*b.y + a.y*b.x)*i
 */
const complexMul = (a: string, b: string) =>
    `vec2<f32>(${a}.x * ${b}.x - ${a}.y * ${b}.y, ${a}.x * ${b}.y + ${a}.y * ${b}.x)`;

/**
 * Complex division: (a.x + a.y*i) / (b.x + b.y*i)
 * = ((a.x*b.x + a.y*b.y) + (a.y*b.x - a.x*b.y)*i) / (b.x^2 + b.y^2)
 */
const complexDiv = (a: string, b: string) => {
    const denom = `(${b}.x * ${b}.x + ${b}.y * ${b}.y)`;
    const realPart = `(${a}.x * ${b}.x + ${a}.y * ${b}.y)`;
    const imagPart = `(${a}.y * ${b}.x - ${a}.x * ${b}.y)`;
    return `vec2<f32>(${realPart} / ${denom}, ${imagPart} / ${denom})`;
};

/**
 * Complex reciprocal: 1 / (a.x + a.y*i)
 * = (a.x - a.y*i) / (a.x^2 + a.y^2)
 */
const complexReciprocal = (a: string) => {
    const denom = `(${a}.x * ${a}.x + ${a}.y * ${a}.y)`;
    return `vec2<f32>(${a}.x / ${denom}, -${a}.y / ${denom})`;
};

/**
 * Complex square: (a.x + a.y*i)^2
 * = (a.x^2 - a.y^2) + (2*a.x*a.y)*i
 */
const complexSquare = (a: string) =>
    `vec2<f32>(${a}.x * ${a}.x - ${a}.y * ${a}.y, 2.0 * ${a}.x * ${a}.y)`;

/**
 * Complex absolute value (modulus): |a| = sqrt(a.x^2 + a.y^2)
 * Note: Returns f32, not vec2<f32>
 */
const complexAbs = (a: string) =>
    `vec2<f32>(sqrt(${a}.x * ${a}.x + ${a}.y * ${a}.y), 0.0)`;

/**
 * Complex conjugate: conj(a) = a.x - a.y*i
 */
const complexConj = (a: string) =>
    `vec2<f32>(${a}.x, -${a}.y)`;

/**
 * Complex exponential: exp(a) = exp(a.x) * (cos(a.y) + i*sin(a.y))
 */
const complexExp = (a: string) =>
    `vec2<f32>(exp(${a}.x) * cos(${a}.y), exp(${a}.x) * sin(${a}.y))`;

/**
 * Complex natural log: log(a) = log(|a|) + i*arg(a)
 * where |a| = sqrt(a.x^2 + a.y^2), arg(a) = atan2(a.y, a.x)
 */
const complexLog = (a: string) =>
    `vec2<f32>(log(sqrt(${a}.x * ${a}.x + ${a}.y * ${a}.y)), atan2(${a}.y, ${a}.x))`;

/**
 * Complex sqrt: sqrt(a) requires branch cut handling
 * sqrt(a) = sqrt((|a| + a.x) / 2) + i * sign(a.y) * sqrt((|a| - a.x) / 2)
 */
const complexSqrt = (a: string) => {
    // Simplified: works for most cases
    const modulus = `sqrt(${a}.x * ${a}.x + ${a}.y * ${a}.y)`;
    return `vec2<f32>(
        sqrt((${modulus} + ${a}.x) * 0.5),
        sign(${a}.y) * sqrt(max(0.0, (${modulus} - ${a}.x) * 0.5))
    )`;
};

// ============================================================================
// POINTWISE_OPS Registry
// ============================================================================

/**
 * POINTWISE_OPS - 所有 Pointwise 操作的表达式配置
 *
 * 分类:
 * - 类别 1: 无 Scalar 参数
 * - 类别 2: 带默认值的 Scalar 参数
 * - 类别 3: Scalar 变体 (tensor op scalar)
 * - 类别 4: Optional Scalar (使用哨兵值)
 */
export const POINTWISE_OPS: Record<string, ExtendedPointwiseOpConfig> = {
    // ================================================================
    // 类别 0: Nullary (Creation)
    // ================================================================

    'zeros': {
        expr: (i, s, t) => `${t}(0)`,
        supportsComplex: true,
        complexExpr: () => 'vec2<f32>(0.0, 0.0)',
    },
    'ones': {
        expr: (i, s, t) => `${t}(1)`,
        supportsComplex: true,
        complexExpr: () => 'vec2<f32>(1.0, 0.0)',
    },
    'empty': {
        expr: (i, s, t) => `${t}(0)`, // Initialize to 0 for determinism
        supportsComplex: true,
        complexExpr: () => 'vec2<f32>(0.0, 0.0)',
    },
    'full': {
        expr: (i, s, t) => {
            const val = s.fill_value;
            // Ensure val is treated as number for Infinity check
            if ((val as unknown as number) === Infinity || val === 'Infinity') {
                // Return FLT_MAX for floats
                if (t === 'f32') return '3.4028235e38';
                if (t === 'i32') return '2147483647';
                if (t === 'u32') return '4294967295';
                return `${t}(${val})`;
            }
            if ((val as unknown as number) === -Infinity || val === '-Infinity') {
                if (t === 'f32') return '-3.4028235e38';
                if (t === 'i32') return '-2147483648';
                return `${t}(0)`;
            }
            return `${t}(${val})`;
        },
        scalarDefaults: { fill_value: 0 },
        supportsComplex: true,
        complexExpr: (i, s) => {
            const val = s.fill_value;
            let valStr = `${val}`;
            if ((val as unknown as number) === Infinity || val === 'Infinity') valStr = '3.4028235e38';
            if ((val as unknown as number) === -Infinity || val === '-Infinity') valStr = '-3.4028235e38';
            return `vec2<f32>(f32(${valStr}), 0.0)`;
        },
    },

    // ================================================================
    // 类别 1: Unary (无 scalar)
    // ================================================================

    'abs': {
        expr: (i) => `abs(${i[0]})`,
        // Complex abs returns f32 (modulus), handled specially in shaderBuilder
        complexExpr: (i) => complexAbs(i[0]),
        supportsComplex: true,
    },
    'neg': {
        expr: (i) => `-(${i[0]})`,
        // WGSL supports -(vec2) natively
        complexExpr: (i) => `-(${i[0]})`,
        supportsComplex: true,
    },
    'sign': { expr: (i) => `sign(${i[0]})` },
    'sqrt': {
        expr: (i) => `sqrt(${i[0]})`,
        complexExpr: (i) => complexSqrt(i[0]),
        supportsComplex: true,
    },
    'rsqrt': { expr: (i) => `inverseSqrt(${i[0]})` },
    'square': {
        expr: (i) => `(${i[0]} * ${i[0]})`,
        complexExpr: (i) => complexSquare(i[0]),
        supportsComplex: true,
    },
    'exp': {
        expr: (i) => `exp(${i[0]})`,
        complexExpr: (i) => complexExp(i[0]),
        supportsComplex: true,
    },
    'exp2': { expr: (i) => `exp2(${i[0]})` },
    'expm1': { expr: (i) => `(exp(${i[0]}) - 1.0)` },
    'log': {
        expr: (i) => `log(${i[0]})`,
        complexExpr: (i) => complexLog(i[0]),
        supportsComplex: true,
    },
    'log2': { expr: (i) => `log2(${i[0]})` },
    'log10': { expr: (i, _, t) => `(log(${i[0]}) / ${t === 'f32' ? '2.302585' : '2.302585'})` },
    'log1p': { expr: (i) => `log(1.0 + ${i[0]})` },
    'sin': { expr: (i) => `sin(${i[0]})` },
    'cos': { expr: (i) => `cos(${i[0]})` },
    'tan': { expr: (i) => `tan(${i[0]})` },
    'asin': { expr: (i) => `asin(${i[0]})` },
    'acos': { expr: (i) => `acos(${i[0]})` },
    'atan': { expr: (i) => `atan(${i[0]})` },
    'sinh': { expr: (i) => `sinh(${i[0]})` },
    'cosh': { expr: (i) => `cosh(${i[0]})` },
    // tanh with numerical stability: clamp input to [-20, 20] to avoid exp overflow
    // tanh(20) ≈ 1.0, tanh(-20) ≈ -1.0, so clamping doesn't affect the result
    'tanh': { expr: (i) => `tanh(clamp(${i[0]}, -20.0, 20.0))` },
    'asinh': { expr: (i) => `asinh(${i[0]})` },
    'acosh': { expr: (i) => `acosh(${i[0]})` },
    'atanh': { expr: (i) => `atanh(${i[0]})` },
    'floor': { expr: (i) => `floor(${i[0]})` },
    'ceil': { expr: (i) => `ceil(${i[0]})` },
    'round': { expr: (i) => `round(${i[0]})` },
    'trunc': { expr: (i) => `trunc(${i[0]})` },
    'frac': { expr: (i) => `fract(${i[0]})` },
    'reciprocal': {
        expr: (i) => `(1.0 / ${i[0]})`,
        complexExpr: (i) => complexReciprocal(i[0]),
        supportsComplex: true,
    },
    /**
     * i0 - Modified Bessel function of the first kind, order 0
     * Uses Abramowitz & Stegun polynomial approximation
     * Requires helper function in shader
     */
    'i0': {
        expr: (i) => `bessel_i0(${i[0]})`,
        helperFunctions: [`
fn bessel_i0(x: f32) -> f32 {
    let ax = abs(x);
    var ans: f32;
    if (ax < 3.75) {
        let y = (x / 3.75) * (x / 3.75);
        ans = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
            + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))));
    } else {
        let y = 3.75 / ax;
        ans = (exp(ax) / sqrt(ax)) * (0.39894228 + y * (0.1328592e-1
            + y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2
            + y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1
            + y * 0.392377e-2))))))));
    }
    return ans;
}
`],
    },
    /**
     * sinc - Normalized sinc function
     * sinc(x) = sin(πx) / (πx), returns 1 when x = 0
     */
    'sinc': {
        expr: (i) => `sinc_impl(${i[0]})`,
        helperFunctions: [`
fn sinc_impl(x: f32) -> f32 {
    if (abs(x) < 1e-9) { return 1.0; }
    let pi_x = ${WGSL_CONSTANTS.PI} * x;
    return sin(pi_x) / pi_x;
}
`],
    },

    // Complex-specific operations
    'conj': {
        expr: (i) => i[0], // For real numbers, conj(x) = x
        complexExpr: (i) => complexConj(i[0]),
        supportsComplex: true,
    },
    'real': {
        expr: (i) => i[0], // For real numbers, real(x) = x
        complexExpr: (i) => `vec2<f32>(${i[0]}.x, 0.0)`, // Returns vec2 (real part is result)
        supportsComplex: true,
    },
    'imag': {
        expr: (i) => '0.0', // For real numbers, imag(x) = 0
        complexExpr: (i) => `vec2<f32>(${i[0]}.y, 0.0)`, // Returns vec2 (real part is result)
        supportsComplex: true,
    },
    /**
     * angle - Complex phase angle
     * For complex: atan2(imag, real)
     * For real: 0 for non-negative, π for negative
     */
    'angle': {
        expr: (i) => `select(0.0, ${WGSL_CONSTANTS.PI}, ${i[0]} < 0.0)`,
        complexExpr: (i) => `vec2<f32>(atan2(${i[0]}.y, ${i[0]}.x), 0.0)`,
        supportsComplex: true,
    },

    // Activation functions (real only)
    'relu': { expr: (i, _, t) => `max(${i[0]}, ${t}(0))` },
    // Sigmoid with numerical stability: clamp input to [-88, 88] to prevent exp overflow
    // exp(88) ≈ 1.65e38 (close to FLT_MAX), exp(-88) ≈ 6e-39 (underflows to 0, which is fine)
    'sigmoid': { expr: (i) => `(1.0 / (1.0 + exp(-clamp(${i[0]}, -88.0, 88.0))))` },
    // SiLU (Swish) with numerical stability: same clamp as sigmoid
    'silu': { expr: (i) => `(${i[0]} / (1.0 + exp(-clamp(${i[0]}, -88.0, 88.0))))` },
    // GELU with numerical stability: clamp tanh input to [-20, 20] to avoid overflow
    'gelu': { expr: (i) => `(0.5 * ${i[0]} * (1.0 + tanh(clamp(0.7978845608 * (${i[0]} + 0.044715 * ${i[0]} * ${i[0]} * ${i[0]}), -20.0, 20.0))))` },
    // Softplus with numerical stability: when x > 20, log(1 + exp(x)) ≈ x
    // This avoids exp overflow for large positive values
    'softplus': { expr: (i) => `select(log(1.0 + exp(${i[0]})), ${i[0]}, ${i[0]} > 20.0)` },
    // Mish with numerical stability: combines stable softplus and clamped tanh
    // mish(x) = x * tanh(softplus(x)), where softplus is stabilized
    'mish': { expr: (i) => `(${i[0]} * tanh(clamp(select(log(1.0 + exp(${i[0]})), ${i[0]}, ${i[0]} > 20.0), -20.0, 20.0)))` },

    // Logical
    'logical_not': { expr: (i) => `!${i[0]}`, outputKind: 'bool' },
    'isnan': { expr: (i) => `(${i[0]} != ${i[0]})`, outputKind: 'bool' },
    'isinf': { expr: (i) => `(abs(${i[0]}) == ${WGSL_CONSTANTS.FLT_MAX})`, outputKind: 'bool' },
    'isfinite': { expr: (i) => `(abs(${i[0]}) < ${WGSL_CONSTANTS.FLT_MAX})`, outputKind: 'bool' },

    // Bitwise (for int types)
    'bitwise_not': { expr: (i) => `~${i[0]}` },

    // Copy (identity) - supports complex
    'copy': {
        expr: (i) => i[0],
        complexExpr: (i) => i[0],
        supportsComplex: true,
    },

    // ================================================================
    // 类别 1: Binary (无 scalar)
    // ================================================================

    'mul': {
        expr: (i) => `(${i[0]} * ${i[1]})`,
        complexExpr: (i) => complexMul(i[0], i[1]),
        supportsComplex: true,
    },
    'div': {
        expr: (i) => `(${i[0]} / ${i[1]})`,
        complexExpr: (i) => complexDiv(i[0], i[1]),
        supportsComplex: true,
    },
    'truediv': {
        expr: (i) => `(${i[0]} / ${i[1]})`,
        complexExpr: (i) => complexDiv(i[0], i[1]),
        supportsComplex: true,
    },
    'floordiv': { expr: (i) => `floor(${i[0]} / ${i[1]})` },
    'fmod': { expr: (i) => `(${i[0]} - trunc(${i[0]} / ${i[1]}) * ${i[1]})` },
    'fmod_scalar': {
        expr: (i, s) => `(${i[0]} - trunc(${i[0]} / ${s.other}) * ${s.other})`,
        scalarDefaults: { other: 1 },
    },
    'remainder': { expr: (i) => `(${i[0]} - floor(${i[0]} / ${i[1]}) * ${i[1]})` },
    'remainder_scalar': {
        expr: (i, s) => `(${i[0]} - floor(${i[0]} / ${s.other}) * ${s.other})`,
        scalarDefaults: { other: 1 },
    },
    'maximum': { expr: (i) => `max(${i[0]}, ${i[1]})` },
    'minimum': { expr: (i) => `min(${i[0]}, ${i[1]})` },
    /**
     * Binary pow: x^y
     * WGSL pow(base, exp) 对负数底数返回 NaN
     * 使用安全公式: select(pow(abs(x), y), sign(x) * pow(abs(x), y), fract(y * 0.5) != 0.0)
     */
    'pow': {
        expr: (i) => {
            const x = i[0];
            const y = i[1];
            const absBase = `abs(${x})`;
            const signBase = `sign(${x})`;
            const powExpr = `pow(${absBase}, ${y})`;
            const isOddExpr = `(fract(${y} * 0.5) != 0.0)`;
            // 奇数指数: sign(x) * pow(abs(x), y); 偶数指数: pow(abs(x), y)
            return `select(${powExpr}, ${signBase} * ${powExpr}, ${isOddExpr})`;
        },
    },
    'atan2': { expr: (i) => `atan2(${i[0]}, ${i[1]})` },
    'hypot': { expr: (i) => `sqrt(${i[0]} * ${i[0]} + ${i[1]} * ${i[1]})` },
    'copysign': { expr: (i) => `copysign(${i[0]}, ${i[1]})` },
    'ldexp': { expr: (i) => `ldexp(${i[0]}, ${i[1]})` },

    // Bitwise (for int types)
    'bitwise_and': { expr: (i) => `(${i[0]} & ${i[1]})` },
    'bitwise_or': { expr: (i) => `(${i[0]} | ${i[1]})` },
    'bitwise_xor': { expr: (i) => `(${i[0]} ^ ${i[1]})` },
    'left_shift': { expr: (i) => `(${i[0]} << ${i[1]})` },
    'right_shift': { expr: (i) => `(${i[0]} >> ${i[1]})` },

    // Logical binary
    'logical_and': { expr: (i) => `(${i[0]} && ${i[1]})`, outputKind: 'bool' },
    'logical_or': { expr: (i) => `(${i[0]} || ${i[1]})`, outputKind: 'bool' },
    'logical_xor': { expr: (i) => `((${i[0]} || ${i[1]}) && !(${i[0]} && ${i[1]}))`, outputKind: 'bool' },

    // ================================================================
    // 类别 1: Comparison (无 scalar, 返回 bool)
    // ================================================================

    'eq': {
        expr: (i) => `(${i[0]} == ${i[1]})`,
        // Complex equality: both components must match
        complexExpr: (i) => `(${i[0]}.x == ${i[1]}.x && ${i[0]}.y == ${i[1]}.y)`,
        outputKind: 'bool',
        supportsComplex: true,
    },
    'ne': {
        expr: (i) => `(${i[0]} != ${i[1]})`,
        complexExpr: (i) => `(${i[0]}.x != ${i[1]}.x || ${i[0]}.y != ${i[1]}.y)`,
        outputKind: 'bool',
        supportsComplex: true,
    },
    'lt': { expr: (i) => `(${i[0]} < ${i[1]})`, outputKind: 'bool' },
    'le': { expr: (i) => `(${i[0]} <= ${i[1]})`, outputKind: 'bool' },
    'gt': { expr: (i) => `(${i[0]} > ${i[1]})`, outputKind: 'bool' },
    'ge': { expr: (i) => `(${i[0]} >= ${i[1]})`, outputKind: 'bool' },

    // ================================================================
    // 类别 1: Ternary (无 scalar)
    // ================================================================

    'where': { expr: (i) => `select(${i[2]}, ${i[1]}, ${i[0]} != 0.0)` },
    'lerp': {
        expr: (i) => `mix(${i[0]}, ${i[1]}, ${i[2]})`,
        // WGSL mix() works with vec2
        complexExpr: (i) => `mix(${i[0]}, ${i[1]}, ${i[2]})`,
        supportsComplex: true,
    },
    'addcmul': {
        expr: (i, s) => `(${i[0]} + ${s.value ?? '1.0'} * ${i[1]} * ${i[2]})`,
        scalarDefaults: { value: 1 },
    },
    'addcdiv': {
        expr: (i, s) => `(${i[0]} + ${s.value ?? '1.0'} * ${i[1]} / ${i[2]})`,
        scalarDefaults: { value: 1 },
    },

    // ================================================================
    // 类别 2: Binary + alpha
    // ================================================================

    'add': {
        expr: (i, s) => `(${i[0]} + ${s.alpha} * ${i[1]})`,
        // For complex: WGSL vec2 addition is component-wise
        // But with alpha scaling, need to handle properly
        complexExpr: (i, s) => `(${i[0]} + vec2<f32>(${s.alpha}) * ${i[1]})`,
        scalarDefaults: { alpha: 1 },
        supportsComplex: true,
    },
    'sub': {
        expr: (i, s) => `(${i[0]} - ${s.alpha} * ${i[1]})`,
        complexExpr: (i, s) => `(${i[0]} - vec2<f32>(${s.alpha}) * ${i[1]})`,
        scalarDefaults: { alpha: 1 },
        supportsComplex: true,
    },

    // ================================================================
    // 类别 3: Scalar 变体 (tensor op scalar)
    // ================================================================

    'add_scalar': {
        expr: (i, s) => `(${i[0]} + ${s.alpha} * ${s.other})`,
        // For complex: scalar is broadcast to (scalar, 0)
        complexExpr: (i, s) => `(${i[0]} + vec2<f32>(f32(${s.alpha}) * f32(${s.other}), 0.0))`,
        scalarDefaults: { alpha: 1, other: 0 },
        supportsComplex: true,
    },
    'sub_scalar': {
        expr: (i, s) => `(${i[0]} - ${s.alpha} * ${s.other})`,
        complexExpr: (i, s) => `(${i[0]} - vec2<f32>(f32(${s.alpha}) * f32(${s.other}), 0.0))`,
        scalarDefaults: { alpha: 1, other: 0 },
        supportsComplex: true,
    },
    'mul_scalar': {
        expr: (i, s) => `(${i[0]} * ${s.other})`,
        // Scalar multiplication broadcasts to both components
        complexExpr: (i, s) => `(${i[0]} * f32(${s.other}))`,
        scalarDefaults: { other: 1 },
        supportsComplex: true,
    },
    'div_scalar': {
        expr: (i, s) => `(${i[0]} / ${s.other})`,
        complexExpr: (i, s) => `(${i[0]} / f32(${s.other}))`,
        scalarDefaults: { other: 1 },
        supportsComplex: true,
    },
    'pow_scalar': {
        /**
         * WGSL pow(base, exp) 对负数底数返回 NaN (undefined behavior)
         *
         * 分两种情况处理：
         * 1. s.exponent 是数字 -> 编译时优化，直接展开
         * 2. s.exponent 是字符串 (uniform 引用) -> 生成运行时安全的 WGSL 代码
         *
         * 安全公式:
         * - 偶数指数: pow(abs(x), n)
         * - 奇数指数: sign(x) * pow(abs(x), n)
         * - 运行时: sign(x)^n * pow(abs(x), n) 需要动态判断
         *
         * 运行时公式 (处理整数n):
         * select(pow(abs(x), n), sign(x) * pow(abs(x), n), fract(n * 0.5) != 0.0)
         * 其中 fract(n*0.5) != 0 判断 n 是否为奇数
         */
        expr: (i, s, t) => {
            const expVal = s.exponent;

            // 如果 exponent 是字符串（uniform 引用），说明是运行时求值
            if (typeof expVal === 'string') {
                // 生成运行时安全的 WGSL 代码
                // 首先处理 n = 0 的情况: x^0 = 1
                const absBase = `abs(${i[0]})`;
                const signBase = `sign(${i[0]})`;
                const powExpr = `pow(${absBase}, ${expVal})`;
                const isOddExpr = `(fract(${expVal} * 0.5) != 0.0)`;
                const isZeroExpr = `(${expVal} == 0.0)`;
                // x^0 = 1; 奇数: sign(x) * pow(abs(x), n); 偶数: pow(abs(x), n)
                const nonZeroPow = `select(${powExpr}, ${signBase} * ${powExpr}, ${isOddExpr})`;
                return `select(${nonZeroPow}, ${t}(1), ${isZeroExpr})`;
            }

            // exponent 是数字，编译时优化
            const exp = Number(expVal ?? 1);

            // x^0 = 1 (包括 0^0 = 1 在数值计算惯例中)
            if (exp === 0) {
                return `${t}(1)`;
            }

            // 常见的整数指数特判
            if (exp === 2) {
                // x^2 = x * x (对负数正确)
                return `(${i[0]} * ${i[0]})`;
            }
            if (exp === -1) {
                return `(1.0 / ${i[0]})`;
            }
            if (exp === -2) {
                return `(1.0 / (${i[0]} * ${i[0]}))`;
            }
            if (exp === 0.5) {
                return `sqrt(${i[0]})`;
            }
            if (exp === -0.5) {
                return `inverseSqrt(${i[0]})`;
            }
            // 检查是否为整数指数
            if (Number.isInteger(exp)) {
                const n = Math.abs(exp);
                const isOdd = n % 2 === 1;
                const isNegExp = exp < 0;
                // 偶数: pow(abs(x), n)
                // 奇数: sign(x) * pow(abs(x), n)
                const baseExpr = isOdd
                    ? `(sign(${i[0]}) * pow(abs(${i[0]}), ${n}.0))`
                    : `pow(abs(${i[0]}), ${n}.0)`;
                return isNegExp ? `(1.0 / ${baseExpr})` : baseExpr;
            }
            // 非整数指数 - 警告：对负数底数会产生 NaN
            // 这是数学上的限制 (负数的非整数次幂是复数)
            return `pow(${i[0]}, f32(${exp}))`;
        },
        scalarDefaults: { exponent: 1 },
    },
    'rsub_scalar': {
        expr: (i, s) => `(${s.other} - ${i[0]})`,
        complexExpr: (i, s) => `(vec2<f32>(f32(${s.other}), 0.0) - ${i[0]})`,
        scalarDefaults: { other: 0 },
        supportsComplex: true,
    },
    'rdiv_scalar': {
        expr: (i, s) => `(${s.other} / ${i[0]})`,
        // scalar / complex = scalar * conj(z) / |z|^2
        complexExpr: (i, s) => {
            const denom = `(${i[0]}.x * ${i[0]}.x + ${i[0]}.y * ${i[0]}.y)`;
            return `vec2<f32>(f32(${s.other}) * ${i[0]}.x / ${denom}, -f32(${s.other}) * ${i[0]}.y / ${denom})`;
        },
        scalarDefaults: { other: 1 },
        supportsComplex: true,
    },

    // Comparison scalar variants
    'eq_scalar': { expr: (i, s) => `(${i[0]} == ${s.other})`, outputKind: 'bool' },
    'ne_scalar': { expr: (i, s) => `(${i[0]} != ${s.other})`, outputKind: 'bool' },
    'lt_scalar': { expr: (i, s) => `(${i[0]} < ${s.other})`, outputKind: 'bool' },
    'le_scalar': { expr: (i, s) => `(${i[0]} <= ${s.other})`, outputKind: 'bool' },
    'gt_scalar': { expr: (i, s) => `(${i[0]} > ${s.other})`, outputKind: 'bool' },
    'ge_scalar': { expr: (i, s) => `(${i[0]} >= ${s.other})`, outputKind: 'bool' },

    // ================================================================
    // 类别 4: Optional Scalar (使用哨兵值)
    // ================================================================

    // NaN-preserving clamp: PyTorch preserves NaN, WGSL clamp() does not
    // Note: WGSL doesn't have isnan(), use (x != x) to check for NaN
    // Formula: select(clamp(x, min, max), x, x != x)
    'clamp': {
        expr: (i, s) => `select(clamp(${i[0]}, ${s.min}, ${s.max}), ${i[0]}, ${i[0]} != ${i[0]})`,
        scalarSentinels: { min: -1e38, max: 1e38 },  // 哨兵值
    },
    'clamp_min': {
        expr: (i, s) => `select(max(${i[0]}, ${s.min}), ${i[0]}, ${i[0]} != ${i[0]})`,
        scalarSentinels: { min: -1e38 },
    },
    'clamp_max': {
        expr: (i, s) => `select(min(${i[0]}, ${s.max}), ${i[0]}, ${i[0]} != ${i[0]})`,
        scalarSentinels: { max: 1e38 },
    },

    // Leaky ReLU
    'leaky_relu': {
        expr: (i, s) => `select(${i[0]}, ${s.negative_slope} * ${i[0]}, ${i[0]} < 0.0)`,
        scalarDefaults: { negative_slope: 0.01 },
    },

    // Hard activation functions with configurable thresholds
    // NaN-preserving: use (x != x) pattern for NaN check
    'hardtanh': {
        expr: (i, s) => `select(clamp(${i[0]}, ${s.min_val}, ${s.max_val}), ${i[0]}, ${i[0]} != ${i[0]})`,
        scalarDefaults: { min_val: -1, max_val: 1 },
    },
    'hardsigmoid': {
        expr: (i) => `max(0.0, min(1.0, ${i[0]} * 0.16666666666666666 + 0.5))`,
    },
    'hardswish': {
        expr: (i) => `(${i[0]} * max(0.0, min(1.0, ${i[0]} * 0.16666666666666666 + 0.5)))`,
    },

    // ELU family - with numerical stability: clamp exp input to prevent underflow
    // When x is very negative (< -88), exp(x) underflows to 0, result is -alpha (which is correct)
    // But we clamp for consistency and to handle edge cases
    'elu': {
        expr: (i, s) => `select(${i[0]}, ${s.alpha} * (exp(clamp(${i[0]}, -88.0, 0.0)) - 1.0), ${i[0]} < 0.0)`,
        scalarDefaults: { alpha: 1.0 },
    },
    'celu': {
        expr: (i, s) => `max(${i[0]}, 0.0) + min(0.0, ${s.alpha} * (exp(clamp(${i[0]} / ${s.alpha}, -88.0, 0.0)) - 1.0))`,
        scalarDefaults: { alpha: 1.0 },
    },
    'selu': {
        expr: (i) => `select(1.0507009873554804934193349852946 * ${i[0]}, 1.0507009873554804934193349852946 * 1.6732632423543772848170429916717 * (exp(clamp(${i[0]}, -88.0, 0.0)) - 1.0), ${i[0]} < 0.0)`,
    },

    /**
     * LogSigmoid: log(sigmoid(x)) = log(1 / (1 + exp(-x)))
     *
     * 数值稳定版本:
     * logsigmoid(x) = -max(-x, 0) - log(1 + exp(-abs(x)))
     *
     * 这个公式在以下情况都保持稳定:
     * - x >> 0: exp(-abs(x)) ≈ 0, 结果 ≈ 0
     * - x << 0: max(-x, 0) = -x, exp(-abs(x)) = exp(x), 结果 ≈ x
     */
    'logsigmoid': {
        expr: (i) => `(-max(-(${i[0]}), 0.0) - log(1.0 + exp(-abs(${i[0]}))))`,
    },

    // Threshold
    'threshold': {
        expr: (i, s) => `select(${i[0]}, ${s.value}, ${i[0]} <= ${s.threshold})`,
        scalarDefaults: { threshold: 0, value: 0 },
    },

    // RReLU (at inference, uses midpoint of lower and upper)
    'rrelu': {
        expr: (i, s) => `select(${i[0]}, (${s.lower} + ${s.upper}) * 0.5 * ${i[0]}, ${i[0]} < 0.0)`,
        scalarDefaults: { lower: 0.125, upper: 0.3333333333333333 },
    },
};

/**
 * 获取操作的输入数量
 */
export function getOpArity(dispatchKey: string): number {
    // Nullary operations
    const nullaryOps = new Set([
        'zeros', 'ones', 'empty', 'full',
    ]);

    // Unary operations
    const unaryOps = new Set([
        'abs', 'neg', 'sign', 'sqrt', 'rsqrt', 'square', 'exp', 'exp2', 'expm1',
        'log', 'log2', 'log10', 'log1p', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
        'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'floor', 'ceil', 'round',
        'trunc', 'frac', 'reciprocal', 'relu', 'sigmoid', 'silu', 'gelu', 'softplus', 'logsigmoid',
        'mish', 'logical_not', 'isnan', 'isinf', 'isfinite', 'bitwise_not', 'copy',
        'leaky_relu', 'hardtanh', 'hardsigmoid', 'hardswish', 'elu', 'celu', 'selu',
        'threshold', 'rrelu', 'clamp', 'clamp_min', 'clamp_max',
        // Complex-specific
        'conj', 'real', 'imag', 'angle',
        // Special math functions
        'i0', 'sinc',
        // Scalar variants are unary
        'add_scalar', 'sub_scalar', 'mul_scalar', 'div_scalar', 'pow_scalar',
        'rsub_scalar', 'rdiv_scalar',
        'eq_scalar', 'ne_scalar', 'lt_scalar', 'le_scalar', 'gt_scalar', 'ge_scalar',
    ]);

    // Ternary operations
    const ternaryOps = new Set([
        'where', 'lerp', 'addcmul', 'addcdiv',
    ]);

    if (nullaryOps.has(dispatchKey)) return 0;
    if (unaryOps.has(dispatchKey)) return 1;
    if (ternaryOps.has(dispatchKey)) return 3;
    return 2; // Default: binary
}

/**
 * Check if an operation supports complex numbers
 */
export function supportsComplexOp(dispatchKey: string): boolean {
    const opConfig = POINTWISE_OPS[dispatchKey];
    return opConfig?.supportsComplex === true || opConfig?.complexExpr !== undefined;
}
