/**
 * NN-Kit Operator Schema v7 - Attention Operations
 *
 * 注意力机制相关操作
 * Mechanism: Composite (组合现有 ops 实现)
 *
 * @module v7/ops/attention
 */

import { SchemaT, SchemaShape, SchemaDtype } from '../helpers';
import type { OpEntry } from '../types';

// ============================================================================
// scaledDotProductAttention (SDPA)
// ============================================================================

/**
 * Scaled Dot-Product Attention
 *
 * 对标 PyTorch torch.nn.functional.scaled_dot_product_attention
 *
 * 计算公式: softmax(Q @ K^T / scale + attn_mask) @ V
 *
 * 输入张量形状:
 * - query: (..., L, E)  - L: target sequence length, E: embedding dimension
 * - key:   (..., S, E)  - S: source sequence length
 * - value: (..., S, Ev) - Ev: value embedding dimension
 *
 * 输出形状: (..., L, Ev)
 */
export const scaledDotProductAttention: OpEntry = {
    name: 'scaledDotProductAttention',
    mechanism: 'Composite',
    signature: {
        params: [
            {
                name: 'query',
                type: SchemaT.Tensor({ ndim: { min: 2 } }),
                doc: 'Query tensor (..., L, E)',
            },
            {
                name: 'key',
                type: SchemaT.Tensor({ ndim: { min: 2 } }),
                doc: 'Key tensor (..., S, E)',
            },
            {
                name: 'value',
                type: SchemaT.Tensor({ ndim: { min: 2 } }),
                doc: 'Value tensor (..., S, Ev)',
            },
            {
                name: 'attnMask',
                type: SchemaT.Optional(SchemaT.Tensor()),
                doc: 'Attention mask (additive). Shape: (..., L, S) or (L, S). Float or Bool type.',
            },
            {
                name: 'dropoutP',
                type: SchemaT.Scalar(),
                default: 0.0,
                doc: 'Dropout probability (ignored in eval mode, not implemented yet)',
            },
            {
                name: 'isCausal',
                type: SchemaT.Bool(),
                default: false,
                doc: 'If true, apply causal mask (upper triangular mask)',
            },
            {
                name: 'scale',
                type: SchemaT.Optional(SchemaT.Scalar()),
                doc: 'Scaling factor. Default: 1/sqrt(E)',
            },
        ],
        returns: { single: SchemaT.Tensor() },
    },
    shape: SchemaShape.explicit('sdpa_output_shape(query, value)'),
    dtype: SchemaDtype.promote(['query', 'key', 'value']),
    dispatchKey: 'scaled_dot_product_attention',
    doc: 'Scaled Dot-Product Attention: softmax(Q @ K^T / scale + mask) @ V',
    codegen: {
        tensorMethod: false,
        namespace: 'nn.functional',
    },
};
