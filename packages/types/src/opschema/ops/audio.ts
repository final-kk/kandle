/**
 * NN-Kit Operator Schema v7 - Audio Operations
 *
 * 音频处理操作
 * Mechanism: IIR (自定义)
 *
 * @module v7/ops/audio
 */

import { SchemaT, SchemaShape, SchemaDtype } from '../helpers';
import type { OpEntry } from '../types';

// ============================================================================
// lfilter_biquad - Biquad IIR 滤波器
// ============================================================================

/**
 * lfilter_biquad - 二阶 IIR 滤波器 (biquad)
 *
 * 直接接收标量系数，避免 Tensor 转换开销。
 *
 * 差分方程:
 * y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
 *
 * 注: a0 归一化为 1 (调用方需预先 normalize)
 *
 * @see https://en.wikipedia.org/wiki/Digital_biquad_filter
 */
export const lfilterBiquad: OpEntry = {
    name: 'lfilterBiquad',
    mechanism: 'Factory',  // 使用 Factory 机制，由 handler 自己创建输出 tensor
    signature: {
        params: [
            { name: 'waveform', type: SchemaT.Tensor({ dtype: 'Floating' }), doc: '输入波形 (..., time)' },
            { name: 'b0', type: SchemaT.Scalar('float'), doc: '分子系数 x[n]' },
            { name: 'b1', type: SchemaT.Scalar('float'), doc: '分子系数 x[n-1]' },
            { name: 'b2', type: SchemaT.Scalar('float'), doc: '分子系数 x[n-2]' },
            { name: 'a1', type: SchemaT.Scalar('float'), doc: '分母系数 y[n-1] (归一化后)' },
            { name: 'a2', type: SchemaT.Scalar('float'), doc: '分母系数 y[n-2] (归一化后)' },
            { name: 'clamp', type: SchemaT.Bool(), default: true, doc: '是否限制输出到 [-1, 1]' },
        ],
        returns: { single: SchemaT.Tensor({ dtype: 'Floating' }) },
    },
    shape: SchemaShape.same('waveform'),
    dtype: SchemaDtype.same('waveform'),
    dispatchKey: 'iir.biquad',
    doc: 'Biquad IIR 滤波器，用于音频信号处理',
    codegen: { tensorMethod: false, staticMethod: false },
};
