/**
 * v5 CodeGen - Internal Function Generator
 *
 * 生成每个 OpEntry 的内部实现函数
 * 每个变体是独立的文件: internal/add_Tensor.ts, internal/add_Scalar.ts
 *
 * v5 重大变更: 生成真正的 Handler 调用而非 stub
 */

import * as fs from 'fs';
import * as path from 'path';
import type { OpEntry, ValueType, ReturnDef, ParamDef } from '../../packages/types/src/opschema/types';

export interface InternalGenStats {
    count: number;
}

/**
 * Generate all internal/*.ts files
 */
export function generateInternalFiles(entries: readonly OpEntry[], outputDir: string): InternalGenStats {
    let count = 0;

    for (const entry of entries) {
        const code = generateInternalFunction(entry);
        const fileName = getInternalFileName(entry);
        fs.writeFileSync(path.join(outputDir, fileName), code);
        count++;
    }

    return { count };
}

/**
 * Get the file name for an OpEntry
 */
function getInternalFileName(entry: OpEntry): string {
    const funcName = entry.variant ? `${entry.name}_${entry.variant}` : entry.name;
    return `${funcName}.ts`;
}

/**
 * Get the function name for an OpEntry
 */
export function getInternalFuncName(entry: OpEntry): string {
    return entry.variant ? `${entry.name}_${entry.variant}` : entry.name;
}

/**
 * Generate a single internal function file
 */
function generateInternalFunction(entry: OpEntry): string {
    const funcName = getInternalFuncName(entry);
    const lines: string[] = [];

    // Header
    lines.push('/**');
    lines.push(` * v5 Internal: ${entry.name}${entry.variant ? `.${entry.variant}` : ''}`);
    lines.push(` * Mechanism: ${entry.mechanism}`);
    lines.push(` * DispatchKey: ${entry.dispatchKey}`);
    if (entry.doc) {
        lines.push(' *');
        lines.push(` * ${entry.doc}`);
    }
    lines.push(' *');

    lines.push(' */');
    lines.push('');

    // Imports - based on mechanism
    lines.push(generateImports(entry));
    lines.push('');

    // Function signature
    const params = generateParamList(entry.signature.params);
    const returnType = generateReturnType(entry);

    lines.push(`export function ${funcName}(`);
    for (let i = 0; i < params.length; i++) {
        const comma = i < params.length - 1 ? ',' : '';
        lines.push(`    ${params[i]}${comma}`);
    }
    lines.push(`): ${returnType} {`);

    // Generate body based on mechanism
    lines.push(generateBody(entry));

    lines.push('}');
    lines.push('');

    return lines.join('\n');
}

/**
 * Generate imports based on mechanism
 */
function generateImports(entry: OpEntry): string {
    const lines: string[] = [];
    lines.push("import type { ITensorHandle } from '@kandle/types';");

    // Import OpEntry definition for this operation - use the opschema.ops namespace from @kandle/types
    lines.push(`import { opschema } from '@kandle/types';`);
    lines.push(`const __entry = opschema.ops.${getInternalFuncName(entry)};`);

    // Import handler based on mechanism
    switch (entry.mechanism) {
        case 'Iterator':
            lines.push("import { dispatchIterator, type OperatorContext } from '../../dispatch/handlers';");
            break;
        case 'Composite':
            lines.push("import { dispatchComposite, type OperatorContext } from '../../dispatch/handlers';");
            break;
        case 'View':
            lines.push("import { dispatchView, type OperatorContext } from '../../dispatch/handlers';");
            break;
        case 'Factory':
            lines.push("import { dispatchFactory, type OperatorContext } from '../../dispatch/handlers';");
            break;
        case 'Copy':
            lines.push("import { dispatchCopy, type OperatorContext } from '../../dispatch/handlers';");
            break;
        // 专用 Kernel 机制
        case 'Matrix':
            lines.push("import { dispatchMatrix, type OperatorContext } from '../../dispatch/handlers';");
            break;
        case 'Window':
            lines.push("import { dispatchWindow, type OperatorContext } from '../../dispatch/handlers';");
            break;
        case 'Normalize':
            lines.push("import { dispatchNormalize, type OperatorContext } from '../../dispatch/handlers';");
            break;
        case 'Sort':
            lines.push("import { dispatchSort, type OperatorContext } from '../../dispatch/handlers';");
            break;
        case 'Gather':
            lines.push("import { dispatchGather, type OperatorContext } from '../../dispatch/handlers';");
            break;
        case 'Scatter':
            lines.push("import { dispatchScatter, type OperatorContext } from '../../dispatch/handlers';");
            break;
        case 'Triangular':
            lines.push("import { dispatchTriangular, type OperatorContext } from '../../dispatch/handlers';");
            break;
        case 'Shape':
            lines.push("import { dispatchShape, type OperatorContext } from '../../dispatch/handlers';");
            break;
        case 'WindowFunc':
            lines.push("import { dispatchWindowFunc, type OperatorContext } from '../../dispatch/handlers';");
            break;
        case 'FFT':
            lines.push("import { dispatchFFT, type OperatorContext } from '../../dispatch/handlers';");
            break;
        default:
            throw new Error(`Unknown mechanism: ${entry.mechanism}`);
    }

    return lines.join('\n');
}

/**
 * Generate function body based on mechanism
 */
function generateBody(entry: OpEntry): string {
    const isTuple = 'tuple' in entry.signature.returns;

    switch (entry.mechanism) {
        case 'Iterator':
            return generateIteratorBody(entry, isTuple);
        case 'Composite':
            return generateCompositeBody(entry, isTuple);
        case 'View':
            return generateViewBody(entry);
        case 'Factory':
            return generateFactoryBody(entry);
        case 'Copy':
            return generateCopyBody(entry);
        // 专用 Kernel 机制
        case 'Matrix':
            return generateMatrixBody(entry, isTuple);
        case 'Window':
            return generateWindowBody(entry, isTuple);
        case 'Normalize':
            return generateNormalizeBody(entry, isTuple);
        case 'Sort':
            return generateSortBody(entry, isTuple);
        case 'Gather':
            return generateGatherBody(entry, isTuple);
        case 'Scatter':
            return generateScatterBody(entry, isTuple);
        case 'Triangular':
            return generateTriangularBody(entry, isTuple);
        case 'Shape':
            return generateShapeBody(entry, isTuple);
        case 'WindowFunc':
            return generateWindowFuncBody(entry, isTuple);
        case 'FFT':
            return generateFFTBody(entry, isTuple);
        default:
            throw new Error(`Unknown mechanism: ${entry.mechanism}`);
    }
}

/**
 * Generate standard Context building code
 */
function generateContextCode(entry: OpEntry): string {
    const lines: string[] = [];

    // Separate params
    // Tensor params: Tensor or Optional<Tensor>. Except 'out'.
    const tensorParams = entry.signature.params.filter(p =>
        (p.type.kind === 'Tensor' || (p.type.kind === 'Optional' && p.type.inner.kind === 'Tensor')) && p.name !== 'out'
    );

    // Scalar args: Boolean, Number, String (primitives).
    // Note: Some legacy handlers (Reduction) might expect dim: number | number[] in metadata or scalarArgs.
    // In v6, simple scalars go to scalarArgs, complex/arrays go to metadata.
    const scalarParams = entry.signature.params.filter(p => {
        if (p.name === 'out') return false;
        const type = p.type.kind === 'Optional' ? p.type.inner : p.type;
        return type.kind === 'Scalar' || type.kind === 'Bool' || type.kind === 'String';
    });

    // Metadata: Everything else (Arrays, Shapes, Axes) + Scalar params that might be used as metadata logic.
    // We put almost everything except tensors into metadata for robustness in handlers.
    // Specifically: Shapes, Axes, TensorList? (TensorList is tensorInputs?)
    // TensorList handling:
    // If param is TensorList, it should be flat-spread into tensorInputs?
    // Current helper `valueTypeToTSInternal` maps TensorList to `ITensorHandle[]`.
    // My context builder below puts `param.name` into `tensorInputs`.
    // If `param.name` is an array of handles, putting it in `[..., param.name]` creates nested array `[..., [h1, h2]]`?
    // `tensorInputs` expects `ITensorHandle[]`.
    // If a param IS `ITensorHandle[]`, we should spread it?
    // `OperatorContext.tensorInputs` is `readonly ITensorHandle[]`.
    // So yes, we need to handle TensorList properly if any op uses it.
    // Currently most ops use Tensor. `cat` uses TensorList.
    // Let's assume for now ops are simple tensors. `cat` might need special handling or is not in this schema subset yet (it's in factory or similar?).

    const metadataParams = entry.signature.params.filter(p => {
        if (p.name === 'out') return false;
        const type = p.type.kind === 'Optional' ? p.type.inner : p.type;
        if (type.kind === 'Tensor') return false;
        // Exclude simple scalars if we want strictly separation, but including them in metadata is harmless and often useful.
        // Let's include everything non-Tensor in metadata.
        return true;
    });

    lines.push(`    const ctx: OperatorContext = {`);
    lines.push(`        opName: '${entry.dispatchKey}',`);

    // Tensor Inputs construction
    if (tensorParams.length > 0) {
        // We need to handle `undefined` for optional tensors.
        // And we need to handle proper comma separation.
        const inputList = tensorParams.map(p => p.name).join(', ');

        // Check for Optional tensors
        const hasOptional = tensorParams.some(p => p.type.kind === 'Optional');
        if (hasOptional) {
            lines.push(`        tensorInputs: [${inputList}].filter((t): t is ITensorHandle => t !== undefined),`);
        } else {
            lines.push(`        tensorInputs: [${inputList}],`);
        }
    } else {
        lines.push(`        tensorInputs: [],`);
    }

    // Scalar Args
    if (scalarParams.length > 0) {
        const entries = scalarParams.map(p => p.name).join(', ');
        lines.push(`        scalarArgs: { ${entries} } as Record<string, any>,`);
    } else {
        lines.push(`        scalarArgs: {},`);
    }

    // Metadata
    if (metadataParams.length > 0) {
        const entries = metadataParams.map(p => p.name).join(', ');
        lines.push(`        metadata: { ${entries} },`);
    } else {
        lines.push(`        metadata: {},`);
    }

    // Outs
    const outParam = entry.signature.params.find(p => p.name === 'out');
    if (outParam) {
        lines.push(`        ...(out !== undefined ? { outs: [out] } : {}),`);
    }

    lines.push(`    };`);
    return lines.join('\n');
}

function generateIteratorBody(entry: OpEntry, isTuple: boolean): string {
    const lines: string[] = [];
    lines.push(generateContextCode(entry));
    if (isTuple) {
        lines.push(`    return dispatchIterator(__entry, ctx) as [ITensorHandle, ITensorHandle];`);
    } else {
        lines.push(`    return dispatchIterator(__entry, ctx) as ITensorHandle;`);
    }
    return lines.join('\n');
}

function generateCompositeBody(entry: OpEntry, isTuple: boolean): string {
    const lines: string[] = [];
    lines.push(generateContextCode(entry));
    if (isTuple) {
        lines.push(`    return dispatchComposite(__entry, ctx) as [ITensorHandle, ITensorHandle];`);
    } else {
        lines.push(`    return dispatchComposite(__entry, ctx) as ITensorHandle;`);
    }
    return lines.join('\n');
}

function generateViewBody(entry: OpEntry): string {
    const lines: string[] = [];
    lines.push(generateContextCode(entry));
    lines.push(`    return dispatchView(__entry, ctx);`);
    return lines.join('\n');
}

function generateFactoryBody(entry: OpEntry): string {
    const lines: string[] = [];
    lines.push(generateContextCode(entry));
    lines.push(`    return dispatchFactory(__entry, ctx);`);
    return lines.join('\n');
}

function generateCopyBody(entry: OpEntry): string {
    const lines: string[] = [];
    lines.push(generateContextCode(entry));
    lines.push(`    return dispatchCopy(__entry, ctx);`);
    return lines.join('\n');
}

// =============================================================================
// 专用 Kernel 机制的 Body 生成
// =============================================================================

function generateMatrixBody(entry: OpEntry, isTuple: boolean): string {
    const lines: string[] = [];
    lines.push(generateContextCode(entry));
    lines.push(`    return dispatchMatrix(__entry, ctx) as ITensorHandle;`);
    return lines.join('\n');
}

function generateWindowBody(entry: OpEntry, isTuple: boolean): string {
    const lines: string[] = [];
    lines.push(generateContextCode(entry));
    const retType = entry.codegen?.conditionalReturn ? 'ITensorHandle | ITensorHandle[]' : 'ITensorHandle';
    lines.push(`    return dispatchWindow(__entry, ctx) as any;`);
    return lines.join('\n');
}

function generateNormalizeBody(entry: OpEntry, isTuple: boolean): string {
    const lines: string[] = [];
    lines.push(generateContextCode(entry));
    lines.push(`    return dispatchNormalize(__entry, ctx) as ITensorHandle;`);
    return lines.join('\n');
}

function generateSortBody(entry: OpEntry, isTuple: boolean): string {
    const lines: string[] = [];
    lines.push(generateContextCode(entry));
    if (isTuple) {
        lines.push(`    return dispatchSort(__entry, ctx) as any;`);
    } else {
        lines.push(`    return dispatchSort(__entry, ctx) as ITensorHandle;`);
    }
    return lines.join('\n');
}

function generateGatherBody(entry: OpEntry, isTuple: boolean): string {
    const lines: string[] = [];
    lines.push(generateContextCode(entry));
    lines.push(`    return dispatchGather(__entry, ctx) as ITensorHandle;`);
    return lines.join('\n');
}

function generateScatterBody(entry: OpEntry, isTuple: boolean): string {
    const lines: string[] = [];
    lines.push(generateContextCode(entry));
    lines.push(`    return dispatchScatter(__entry, ctx) as ITensorHandle;`);
    return lines.join('\n');
}

function generateTriangularBody(entry: OpEntry, isTuple: boolean): string {
    const lines: string[] = [];
    lines.push(generateContextCode(entry));
    lines.push(`    return dispatchTriangular(__entry, ctx) as ITensorHandle;`);
    return lines.join('\n');
}

function generateShapeBody(entry: OpEntry, isTuple: boolean): string {
    const lines: string[] = [];
    lines.push(generateContextCode(entry));
    lines.push(`    return dispatchShape(__entry, ctx) as ITensorHandle;`);
    return lines.join('\n');
}

function generateWindowFuncBody(entry: OpEntry, isTuple: boolean): string {
    const lines: string[] = [];
    lines.push(generateContextCode(entry));
    lines.push(`    return dispatchWindowFunc(__entry, ctx) as ITensorHandle;`);
    return lines.join('\n');
}

function generateFFTBody(entry: OpEntry, isTuple: boolean): string {
    const lines: string[] = [];
    lines.push(generateContextCode(entry));
    lines.push(`    return dispatchFFT(__entry, ctx) as ITensorHandle;`);
    return lines.join('\n');
}

/**
 * Get the ops file path for an entry
 *
 * v7 更新: 使用新的语义领域文件名
 */
function getOpFilePath(entry: OpEntry): string {
    const name = entry.name;

    // Pointwise (合并 unary + arithmetic + comparison)
    // 包含 tanh, sigmoid, relu 作为 torch.xxx 数学函数
    if (['add', 'sub', 'mul', 'div', 'pow', 'fmod', 'remainder', 'maximum', 'minimum', 'floorDivide',
        'abs', 'neg', 'sign', 'sqrt', 'rsqrt', 'square', 'exp', 'exp2', 'expm1', 'log', 'log2', 'log10', 'log1p',
        'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
        'ceil', 'floor', 'round', 'trunc', 'frac',
        'reciprocal', 'erf', 'erfc', 'i0', 'sinc', 'logicalNot', 'clamp', 'conj', 'real', 'imag', 'angle',
        'sigmoid', 'relu',
        'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'isnan', 'isinf', 'isfinite', 'where'].includes(name)) {
        return 'pointwise';
    }

    // Activation (神经网络激活函数)
    // 注意: sigmoid, relu, tanh 已移至 pointwise
    if (['softmax', 'logSoftmax', 'softmin',
        'gelu', 'silu', 'leakyRelu', 'elu', 'hardtanh', 'dropout',
        'logsigmoid', 'selu', 'softplus'].includes(name)) {
        return 'activation';
    }

    // Reduction
    if (['sum', 'mean', 'prod', 'max', 'min', 'argmax', 'argmin', 'all', 'any',
        'variance', 'std', 'norm', 'nanmean', 'nansum', 'logsumexp'].includes(name)) {
        return 'reduction';
    }

    // Linalg (线性代数, 原 matrix)
    if (['matmul', 'mm', 'bmm', 'mv', 'dot', 'addmm', 'addmv', 'outer', 'baddbmm', 'linear'].includes(name)) {
        return 'linalg';
    }

    // Triangular (三角矩阵, 原 matrix_transforms)
    if (['triu', 'tril', 'diagonal', 'diag', 'trace'].includes(name)) {
        return 'triangular';
    }

    // Norm (归一化)
    if (['batchNorm', 'groupNorm', 'layerNorm', 'rmsNorm', 'normalize'].includes(name)) {
        return 'norm';
    }

    // Shape (形状操作, 原 view)
    if (['reshape', 'view', 'permute', 'transpose', 'unsqueeze', 'squeeze', 'flatten', 'expand', 'select', 'slice', 'cat', 'stack', 'repeatInterleave', 'asStrided', 'diff', 'flip', 'fliplr', 'flipud'].includes(name)) {
        return 'shape';
    }

    // Creation (创建操作, 原 factory)
    if (['zeros', 'ones', 'empty', 'full', 'zerosLike', 'onesLike', 'emptyLike', 'arange', 'linspace', 'rand', 'randn', 'randint', 'eye', 'multinomial',
        'hannWindow', 'hammingWindow', 'blackmanWindow', 'bartlettWindow', 'kaiserWindow', 'pad'].includes(name)) {
        return 'creation';
    }

    // Memory (内存操作, 原 copy)
    if (['contiguous', 'clone', 'to', 'cast'].includes(name)) {
        return 'memory';
    }

    // Scan
    if (['cumsum', 'cumprod', 'cummax', 'cummin'].includes(name)) {
        return 'scan';
    }

    // Sort
    if (['sort', 'argsort', 'topk'].includes(name)) {
        return 'sort';
    }

    // Conv/Pool
    if (['conv1d', 'conv2d', 'conv3d', 'convTranspose2d',
        'maxPool1d', 'maxPool2d', 'maxPool3d',
        'avgPool1d', 'avgPool2d', 'avgPool3d',
        'adaptiveAvgPool2d', 'adaptiveMaxPool2d'].includes(name)) {
        return 'conv';
    }

    // Indexing (索引操作, 合并 gather + scatter + embedding)
    if (['indexSelect', 'gather', 'indexAdd', 'scatter', 'scatterAdd', 'scatterReduce', 'embedding'].includes(name)) {
        return 'indexing';
    }

    // Attention (注意力机制)
    if (['scaledDotProductAttention'].includes(name)) {
        return 'attention';
    }

    // FFT (快速傅里叶变换)
    if (['fft', 'ifft', 'rfft', 'irfft', 'fft2', 'ifft2', 'rfft2', 'irfft2', 'fftn', 'ifftn', 'rfftn', 'irfftn', 'hfft', 'ihfft', 'stft', 'istft', 'fftshift', 'ifftshift', 'fftfreq', 'rfftfreq'].includes(name)) {
        return 'fft';
    }

    // Fallback - 默认返回 pointwise
    return 'pointwise';
}

/**
 * Generate parameter list with TypeScript types
 */
function generateParamList(params: readonly ParamDef[]): string[] {
    return params.map(p => {
        const optional = p.default !== undefined || isOptional(p.type) ? '?' : '';
        const tsType = valueTypeToTSInternal(p.type);
        return `${p.name}${optional}: ${tsType}`;
    });
}

function isOptional(type: ValueType): boolean {
    return type.kind === 'Optional';
}

function valueTypeToTSInternal(type: ValueType): string {
    switch (type.kind) {
        case 'Tensor':
            return 'ITensorHandle';
        case 'TensorList':
            return 'ITensorHandle[]';
        case 'Scalar':
            return type.numericKind === 'bool' ? 'boolean' : 'number';
        case 'ScalarList':
            return 'number[]';
        case 'Shape':
            return 'readonly number[]';
        case 'Axis':
            return 'number';
        case 'Axes':
            return 'number | readonly number[]';
        case 'DType':
            return 'string';
        case 'Device':
            return 'string';
        case 'Bool':
            return 'boolean';
        case 'String':
            if (type.oneOf && type.oneOf.length > 0) {
                return type.oneOf.map(s => `'${s}'`).join(' | ');
            }
            return 'string';
        case 'Optional':
            return `${valueTypeToTSInternal(type.inner)} | undefined`;
        case 'Union':
            return type.types.map(valueTypeToTSInternal).join(' | ');
        default:
            return 'unknown';
    }
}

function generateReturnType(entry: OpEntry): string {
    const returns = entry.signature.returns;
    const condRet = entry.codegen?.conditionalReturn;

    if (condRet) {
        const tupleType = Array(condRet.tupleSize).fill('ITensorHandle').join(', ');
        return `ITensorHandle | [${tupleType}]`;
    }

    if ('single' in returns) {
        return valueTypeToTSInternal(returns.single);
    } else {
        const types = returns.tuple.map(t => valueTypeToTSInternal(t.type));
        return `[${types.join(', ')}]`;
    }
}
