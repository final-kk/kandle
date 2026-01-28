/**
 * TensorIterator V2 - 张量操作的迭代抽象
 * 
 * 设计理念：
 * - 一旦构建完成，此对象不可变（immutable）
 * - 所有构建逻辑集中在 build() 方法中
 * - 提供便捷工厂方法简化常见操作
 * - outputs-first 排列：operands 数组中输出在前，输入在后
 */

import {
    DType,
    ITensorHandle,
    ITensorIterator,
    TensorIteratorConfig,
    TensorIteratorOperand,
    DeviceNameEnum,
    Shape,
} from '@kandle/types';

import {
    computeBroadcastShape,
    computeBroadcastStrides,
    isShapeEquals,
    resolveCommonDType,
    resolveComputationType,
    resolveResultType,
    resolveResultTypeWithPolicy,
    Logger,
    detectOverlap,
    isContiguousStrides,
    computeNumel,
    normalizeAxis,
    computeReductionShape,
    computeReductionDimShape,
    splitStridesForReduction,
    computeKeepDimsStrides,
} from '@kandle/utils';

import { env } from '../env';

const logger = new Logger('TensorIterator');

/**
 * TensorIterator - 管理张量操作的迭代逻辑
 * 
 * 使用模式：
 * ```typescript
 * const iter = TensorIterator.binaryOp(a, b, 'add', out);
 * backend.operators.find('add')(iter);
 * return iter.output().tensorHandle;
 * ```
 */
export class TensorIterator implements ITensorIterator {
    // ===========================================
    // 形状信息
    // ===========================================
    readonly inputShape: readonly number[];
    readonly outputShape: readonly number[];
    readonly outputNumel: number;

    // ===========================================
    // 类型信息
    // ===========================================
    readonly commonDtype: DType;
    readonly computeDtype: DType;

    // ===========================================
    // 操作数
    // ===========================================
    readonly operands: readonly TensorIteratorOperand[];
    readonly numOutputs: number;
    readonly numInputs: number;

    // ===========================================
    // Reduction 信息
    // ===========================================
    readonly isReduction: boolean;
    readonly reductionAxes: readonly number[];
    readonly keepDims: boolean;
    readonly reductionShape: readonly number[];
    readonly reductionNumel: number;

    // ===========================================
    // 优化标记
    // ===========================================
    readonly isContiguous: boolean;
    readonly isSafeInplace: boolean;

    // ===========================================
    // v5: Scalar Arguments (OperatorContext)
    // ===========================================
    private _scalarArgs: Record<string, number | boolean | string | number[] | undefined> = {};

    /**
     * 设置标量参数 (由 Handler 调用)
     */
    setScalarArgs(args: Record<string, number | boolean | string | number[] | undefined>): void {
        this._scalarArgs = { ...args };
    }

    /**
     * 获取单个标量参数
     */
    getScalarArg<T extends number | boolean | string | number[]>(name: string): T | undefined {
        return this._scalarArgs[name] as T | undefined;
    }

    /**
     * 获取所有标量参数
     */
    getScalarArgs(): Record<string, number | boolean | string | number[] | undefined> {
        return { ...this._scalarArgs };
    }

    /**
     * 私有构造函数，只能通过 build() 或工厂方法创建
     */
    private constructor(params: {
        inputShape: readonly number[];
        outputShape: readonly number[];
        commonDtype: DType;
        computeDtype: DType;
        operands: TensorIteratorOperand[];
        numOutputs: number;
        isReduction: boolean;
        reductionAxes: readonly number[];
        keepDims: boolean;
        reductionShape: readonly number[];
        isContiguous: boolean;
        isSafeInplace: boolean;
    }) {
        this.inputShape = params.inputShape;
        this.outputShape = params.outputShape;
        this.outputNumel = computeNumel(params.outputShape);
        this.commonDtype = params.commonDtype;
        this.computeDtype = params.computeDtype;
        this.operands = params.operands;
        this.numOutputs = params.numOutputs;
        this.numInputs = params.operands.length - params.numOutputs;
        this.isReduction = params.isReduction;
        this.reductionAxes = params.reductionAxes;
        this.keepDims = params.keepDims;
        this.reductionShape = params.reductionShape;
        this.reductionNumel = computeNumel(params.reductionShape.length > 0 ? params.reductionShape : [1]);
        this.isContiguous = params.isContiguous;
        this.isSafeInplace = params.isSafeInplace;
    }

    // ===========================================
    // 访问器方法
    // ===========================================

    /**
     * 快捷访问第 idx 个输出（默认 0）
     */
    output(idx: number = 0): TensorIteratorOperand {
        if (idx < 0 || idx >= this.numOutputs) {
            throw new Error(`Output index ${idx} out of bounds (numOutputs=${this.numOutputs})`);
        }
        return this.operands[idx];
    }

    /**
     * 快捷访问第 idx 个输入（默认 0）
     */
    input(idx: number = 0): TensorIteratorOperand {
        if (idx < 0 || idx >= this.numInputs) {
            throw new Error(`Input index ${idx} out of bounds (numInputs=${this.numInputs})`);
        }
        return this.operands[this.numOutputs + idx];
    }

    // ===========================================
    // 主构建入口
    // ===========================================

    /**
     * 统一构建入口
     * 
     * 所有构建逻辑集中在此方法中：
     * 1. 验证配置完整性
     * 2. 设备一致性检查
     * 3. 类型计算
     * 4. 形状计算
     * 5. 输出处理
     * 6. 内存重叠检测
     * 7. 构建 operands
     * 8. 计算优化标记
     */
    static build(config: TensorIteratorConfig): TensorIterator {
        const { inputs, outputs, opName, reduction } = config;

        logger.debug(`Building TensorIterator for op '${opName}'`);
        logger.debug(`  inputs: ${inputs.length}, outputs: ${outputs.length}`);

        // ===== 1. 验证配置完整性 =====
        TensorIterator.validateConfig(config);

        // ===== 2. 设备一致性检查 =====
        const device = TensorIterator.computeCommonDevice(config);

        // ===== 3. 类型计算 =====
        const { commonDtype, computeDtype, resultDtype } = TensorIterator.computeTypes(config);
        logger.debug(`  commonDtype: ${commonDtype}, computeDtype: ${computeDtype}, resultDtype: ${resultDtype}`);

        // ===== 4. 形状计算 =====
        let { inputShape, outputShape, reductionShape } = TensorIterator.computeShapes(config);
        logger.debug(`  inputShape: [${inputShape.join(', ')}], outputShape: [${outputShape.join(', ')}]`);
        if (reduction) {
            logger.debug(`  reductionShape: [${reductionShape.join(', ')}]`);
        }

        // ===== 5. 输出处理 =====
        const backend = env.getBackend(device);
        const outputHandles = TensorIterator.handleOutputs(
            config,
            outputShape,
            resultDtype,
            backend
        );

        // ===== 6. 内存重叠检测 =====
        if (config.checkMemOverlap !== false) {
            TensorIterator.checkMemoryOverlap(inputs, outputHandles);
        }

        // ===== 7. 构建 operands =====
        const operands = TensorIterator.buildOperands(
            inputs,
            outputHandles,
            inputShape,
            outputShape,
            reduction
        );

        // ===== 8. 维度折叠优化 (Dimension Coalescing) =====
        // 仅针对 Elementwise 操作进行优化，Reduction 涉及复杂的轴映射暂时跳过
        // 如果设置了 disableDimensionCoalescing，跳过此优化（用于输入输出布局不同的情况）
        if (!reduction && !config.disableDimensionCoalescing) {
            const coalesced = TensorIterator.coalesceDimensions(inputShape as number[], operands);
            if (coalesced) {
                logger.debug(`Dimensions coalesced: [${inputShape.join(', ')}] -> [${coalesced.shape.join(', ')}]`);

                // 更新 shape 和 operands
                // 注意: inputShape 和 outputShape 在 elementwise 下是相同的
                // 我们不可以修改只读属性，所以这里重新赋值局部变量，后续传入构造函数
                // inputShape 是 readonly, 需要 cast
                inputShape = coalesced.shape;
                outputShape = coalesced.shape;

                // operands 已经在 coalesceDimensions 内部被修改了 (in-place modification of arrays inside objects)
                // update operands shape property just in case it is read later
                operands.forEach(op => {
                    op.shape = coalesced.shape;
                });
            }
        }

        // ===== 9. 计算优化标记 =====
        const isContiguous = TensorIterator.computeContiguity(operands, outputShape);
        const isSafeInplace = TensorIterator.computeInplaceSafety(operands);

        return new TensorIterator({
            inputShape,
            outputShape,
            commonDtype,
            computeDtype,
            operands,
            numOutputs: outputs.length,
            isReduction: !!reduction,
            reductionAxes: reduction ? reduction.axes : [],
            keepDims: reduction ? reduction.keepDims : false,
            reductionShape,
            isContiguous,
            isSafeInplace,
        });
    }

    // ===========================================
    // 便捷工厂方法
    // ===========================================

    /**
     * Unary 操作
     */
    static unaryOp(
        input: ITensorHandle,
        opName: TensorIteratorConfig['opName'],
        out?: ITensorHandle
    ): TensorIterator {
        return TensorIterator.build({
            inputs: [input],
            outputs: [out],
            opName,
        });
    }

    /**
     * Binary 操作
     */
    static binaryOp(
        a: ITensorHandle,
        b: ITensorHandle,
        opName: TensorIteratorConfig['opName'],
        out?: ITensorHandle
    ): TensorIterator {
        return TensorIterator.build({
            inputs: [a, b],
            outputs: [out],
            opName,
        });
    }

    /**
     * Ternary 操作
     */
    static ternaryOp(
        a: ITensorHandle,
        b: ITensorHandle,
        c: ITensorHandle,
        opName: TensorIteratorConfig['opName'],
        out?: ITensorHandle
    ): TensorIterator {
        return TensorIterator.build({
            inputs: [a, b, c],
            outputs: [out],
            opName,
        });
    }

    /**
     * Reduction 操作（单输出）
     */
    static reductionOp(
        input: ITensorHandle,
        opName: TensorIteratorConfig['opName'],
        axes: number[],
        keepDims: boolean,
        out?: ITensorHandle
    ): TensorIterator {
        return TensorIterator.build({
            inputs: [input],
            outputs: [out],
            opName,
            reduction: { axes, keepDims },
        });
    }

    /**
     * Reduction 操作（双输出：值 + 索引）
     */
    static reductionWithIndicesOp(
        input: ITensorHandle,
        opName: TensorIteratorConfig['opName'],
        axes: number[],
        keepDims: boolean,
        outValues?: ITensorHandle,
        outIndices?: ITensorHandle
    ): TensorIterator {
        return TensorIterator.build({
            inputs: [input],
            outputs: [outValues, outIndices],
            opName,
            reduction: { axes, keepDims },
            outputDtypes: [undefined, 'int64'],
        });
    }

    /**
     * Scan 操作（单输出: cumsum, cumprod）
     * 
     * Scan 操作沿指定维度执行前缀操作，输出 shape 与输入相同。
     */
    static scanOp(
        input: ITensorHandle,
        opName: TensorIteratorConfig['opName'],
        dim: number,
        out?: ITensorHandle
    ): TensorIterator {
        // Scan 操作不是 reduction - 输出 shape 与输入相同
        // 我们通过 scalarArgs 传递 dim 给 backend
        const iterator = TensorIterator.build({
            inputs: [input],
            outputs: [out],
            opName,
            // 禁用维度合并，因为 scan 需要保持原始维度结构以便正确执行
            disableDimensionCoalescing: true,
        });
        // 规范化 dim
        const rank = input.shape.length;
        const normalizedDim = dim < 0 ? dim + rank : dim;
        iterator.setScalarArgs({ dim: normalizedDim });
        return iterator;
    }

    /**
     * Scan 操作（双输出: cummax, cummin 返回 values + indices）
     */
    static scanWithIndicesOp(
        input: ITensorHandle,
        opName: TensorIteratorConfig['opName'],
        dim: number,
        outValues?: ITensorHandle,
        outIndices?: ITensorHandle
    ): TensorIterator {
        const iterator = TensorIterator.build({
            inputs: [input],
            outputs: [outValues, outIndices],
            opName,
            outputDtypes: [undefined, 'int64'],
            disableDimensionCoalescing: true,
        });
        // 规范化 dim
        const rank = input.shape.length;
        const normalizedDim = dim < 0 ? dim + rank : dim;
        iterator.setScalarArgs({ dim: normalizedDim });
        return iterator;
    }

    /**
     * Nullary 操作（factory）
     */
    static nullaryOp(
        shape: Shape,
        dtype: DType,
        opName: TensorIteratorConfig['opName']
    ): TensorIterator {
        return TensorIterator.build({
            inputs: [],
            outputs: [undefined],
            opName,
            staticShape: shape,
            staticDtype: dtype,
        });
    }

    // ===========================================
    // 私有构建辅助方法
    // ===========================================

    /**
     * 验证配置完整性
     */
    private static validateConfig(config: TensorIteratorConfig): void {
        const { inputs, outputs, opName, staticShape, staticDtype } = config;

        // Nullary 操作必须提供 staticShape 和 staticDtype
        if (inputs.length === 0) {
            if (!staticShape) {
                throw new Error(`Nullary operation '${opName}' requires staticShape`);
            }
            if (!staticDtype) {
                throw new Error(`Nullary operation '${opName}' requires staticDtype`);
            }
        }

        // 至少需要一个输出
        if (outputs.length === 0) {
            throw new Error(`At least one output is required for operation '${opName}'`);
        }
    }

    /**
     * 计算公共设备
     */
    private static computeCommonDevice(config: TensorIteratorConfig): DeviceNameEnum {
        const { inputs, outputs } = config;

        // 收集所有已存在的张量
        const allTensors: ITensorHandle[] = [
            ...inputs,
            ...outputs.filter((o): o is ITensorHandle => o !== undefined)
        ];

        if (allTensors.length === 0) {
            // Nullary 操作使用默认设备
            return env.getDefaultDevice().name;
        }

        const device = allTensors[0].device;

        // 检查设备一致性
        if (config.checkAllSameDevice !== false) {
            for (const t of allTensors) {
                if (t.device !== device) {
                    throw new Error(`Device mismatch: ${t.device} vs ${device}`);
                }
            }
        }

        return device;
    }

    /**
     * 计算类型
     */
    private static computeTypes(config: TensorIteratorConfig): {
        commonDtype: DType;
        computeDtype: DType;
        resultDtype: DType;
    } {
        const { inputs, opName, staticDtype, typePromotionKind } = config;

        // 如果有静态 dtype，直接使用
        if (staticDtype) {
            // 如果提供了 typePromotionKind，使用它来决定 resultDtype
            const resultDtype = typePromotionKind
                ? resolveResultTypeWithPolicy(staticDtype, typePromotionKind, opName)
                : resolveResultType(staticDtype, opName);
            return {
                commonDtype: staticDtype,
                computeDtype: resolveComputationType(staticDtype, opName),
                resultDtype,
            };
        }

        // 从输入推断
        if (inputs.length === 0) {
            throw new Error('Cannot infer dtype without inputs or staticDtype');
        }

        const inputDtypes = inputs.map(t => t.dtype);
        const commonDtype = resolveCommonDType(inputDtypes);
        const computeDtype = resolveComputationType(commonDtype, opName);

        // 优先使用直接传入的 typePromotionKind，避免通过 opName 反查 OpRegistry
        const resultDtype = typePromotionKind
            ? resolveResultTypeWithPolicy(commonDtype, typePromotionKind, opName)
            : resolveResultType(commonDtype, opName);

        return { commonDtype, computeDtype, resultDtype };
    }

    /**
     * 计算形状
     */
    private static computeShapes(config: TensorIteratorConfig): {
        inputShape: readonly number[];
        outputShape: readonly number[];
        reductionShape: readonly number[];
    } {
        const { inputs, staticShape, reduction } = config;

        // Nullary 操作
        if (inputs.length === 0 && staticShape) {
            return {
                inputShape: staticShape,
                outputShape: staticShape,
                reductionShape: [],
            };
        }

        // 计算广播后的输入形状
        let inputShape: readonly number[];
        if (inputs.length === 1) {
            inputShape = inputs[0].shape;
        } else if (inputs.length === 2) {
            inputShape = computeBroadcastShape(inputs[0].shape, inputs[1].shape) as readonly number[];
        } else if (inputs.length > 2) {
            // N-ary 广播
            inputShape = inputs.reduce((acc, t) =>
                computeBroadcastShape(acc, t.shape) as readonly number[],
                inputs[0].shape
            );
        } else {
            throw new Error('No inputs provided');
        }

        // Reduction 操作
        if (reduction) {
            const { axes, keepDims } = reduction;
            const outputShape = computeReductionShape(inputShape, axes, keepDims);
            const reductionShape = computeReductionDimShape(inputShape, axes);
            return {
                inputShape,
                outputShape,
                reductionShape,
            };
        }

        // Elementwise 操作
        return {
            inputShape,
            outputShape: inputShape,
            reductionShape: [],
        };
    }

    /**
     * 处理输出张量
     */
    private static handleOutputs(
        config: TensorIteratorConfig,
        outputShape: readonly number[],
        resultDtype: DType,
        backend: ReturnType<typeof env.getBackend>
    ): ITensorHandle[] {
        const { outputs, outputDtypes, reduction } = config;

        return outputs.map((out, idx) => {
            // 确定此输出的 dtype
            // 优先级: 1. outputDtypes 2. 预分配的 out.dtype 3. resultDtype
            // 这允许 cast 操作传入不同 dtype 的预分配输出
            const dtype = outputDtypes?.[idx] ?? (out?.dtype) ?? resultDtype;

            if (out) {
                // 验证已存在的输出张量

                // 验证 shape 兼容性
                if (!isShapeEquals(out.shape, outputShape as Shape)) {
                    throw new Error(
                        `Output shape mismatch: expected [${outputShape.join(', ')}], got [${out.shape.join(', ')}]`
                    );
                }

                // 注意: 不再验证 dtype 兼容性
                // 因为 cast 等操作需要支持不同的输入/输出 dtype
                // dtype 已经通过 out.dtype 正确设置

                return out;
            } else {
                // 自动分配新张量
                return backend.createTensorHandle(outputShape as Shape, dtype);
            }
        });
    }

    /**
     * 检查内存重叠
     */
    private static checkMemoryOverlap(
        inputs: ITensorHandle[],
        outputs: ITensorHandle[]
    ): void {
        for (const out of outputs) {
            for (const inp of inputs) {
                // 只有共享 storage 时才需要检测
                if (out.storageId === inp.storageId) {
                    if (detectOverlap(out, inp)) {
                        throw new Error(
                            `Unsupported: output tensor overlaps with input. ` +
                            `For element-wise ops on overlapping views, this may cause incorrect results. ` +
                            `Consider using a non-overlapping output tensor.`
                        );
                    }
                    logger.debug(`Output shares storage with input, but no overlap detected (safe in-place).`);
                }
            }
        }
    }

    /**
     * 构建 operands
     */
    private static buildOperands(
        inputs: ITensorHandle[],
        outputs: ITensorHandle[],
        inputShape: readonly number[],
        outputShape: readonly number[],
        reduction?: TensorIteratorConfig['reduction']
    ): TensorIteratorOperand[] {
        const operands: TensorIteratorOperand[] = [];

        // Outputs first
        for (let i = 0; i < outputs.length; i++) {
            const out = outputs[i];

            let outputStrides: readonly number[];
            if (reduction && reduction.keepDims) {
                // keepDims 模式下，reduction 维度的 stride 为 0
                const { parallelStrides } = splitStridesForReduction(
                    inputShape as number[],
                    out.strides,
                    reduction.axes
                );
                outputStrides = computeKeepDimsStrides(inputShape, reduction.axes, parallelStrides);
            } else {
                outputStrides = out.strides;
            }

            operands.push({
                name: `output${i}`,
                tensorHandle: out,
                buffer: out.storage.buffer,
                dtype: out.dtype,
                shape: out.shape,
                strides: outputStrides,
                offset: out.offset,
                isOutput: true,
            });
        }

        // Then inputs
        for (let i = 0; i < inputs.length; i++) {
            const inp = inputs[i];

            let strides: readonly number[];
            let reductionStrides: readonly number[] | undefined;

            if (reduction) {
                // Reduction 操作：分离 parallel 和 reduction strides
                const {
                    parallelStrides,
                    reductionStrides: redStrides
                } = splitStridesForReduction(inp.shape, inp.strides, reduction.axes);
                strides = parallelStrides;
                reductionStrides = redStrides;
            } else {
                // Elementwise 操作：广播 strides
                strides = computeBroadcastStrides(inp.shape, inp.strides, outputShape as Shape);
            }

            operands.push({
                name: `input${i}`,
                tensorHandle: inp,
                buffer: inp.storage.buffer,
                dtype: inp.dtype,
                shape: inp.shape,
                strides,
                offset: inp.offset,
                isOutput: false,
                reductionStrides,
            });
        }

        return operands;
    }

    /**
     * 计算是否连续
     */
    private static computeContiguity(
        operands: TensorIteratorOperand[],
        outputShape: readonly number[]
    ): boolean {
        return operands.every(op => {
            if (op.offset !== 0) return false;
            return isContiguousStrides(outputShape, op.strides);
        });
    }

    /**
     * 计算 in-place 安全性
     */
    private static computeInplaceSafety(operands: TensorIteratorOperand[]): boolean {
        const outputOp = operands.find(op => op.isOutput);
        if (!outputOp) return true;

        const outputBuffer = outputOp.buffer;
        const outputOffset = outputOp.offset;
        const outputStrides = outputOp.strides;

        // Check all input operands
        for (const op of operands) {
            if (op.isOutput) continue;

            // Only care about inputs that share buffer with output
            if (op.buffer !== outputBuffer) continue;

            // Check if geometry is identical
            const sameOffset = op.offset === outputOffset;
            const sameStrides =
                op.strides.length === outputStrides.length &&
                op.strides.every((s, i) => s === outputStrides[i]);

            if (!sameOffset || !sameStrides) {
                // Shares buffer but different geometry -> NOT safe
                return false;
            }
        }

        return true;
    }

    /**
     * 维度折叠 (Dimension Coalescing)
     * 尝试合并相邻的维度，减少 GPU 索引计算开销。
     * 
     * 算法：
     * 从最内层维度开始向外遍历，如果相邻维度 [i, i+1] 满足 contiguous 条件（stride[i] = stride[i+1] * shape[i+1]），
     * 则合并这两个维度。
     * 
     * @returns 如果发生了折叠，返回新的 shape；否则返回 undefined
     */
    private static coalesceDimensions(
        shape: number[],
        operands: TensorIteratorOperand[]
    ): { shape: number[] } | undefined {
        const ndim = shape.length;
        if (ndim <= 1) return undefined;

        const newShape: number[] = [];
        const newStridesList: number[][] = operands.map(() => []);

        // 初始化当前合并块
        let currDimSize = shape[ndim - 1];
        let currStrides = operands.map(op => op.strides[ndim - 1]);

        // 从倒数第二个维度向前遍历
        for (let i = ndim - 2; i >= 0; i--) {
            const prevDimSize = shape[i];
            const prevStrides = operands.map(op => op.strides[i]);

            let canMerge = true;
            for (let k = 0; k < operands.length; k++) {
                const prevStride = prevStrides[k];
                const currStride = currStrides[k];

                // 检查连续性条件: stride[i] == stride[i+1] * shape[i+1]
                // 特殊情况: 如果 dimensions size 为 1，它不影响 stride 连续性 (总是连续)
                if (currDimSize === 1) {
                    // 如果内层是 1，外层 stride 必须匹配... 
                    // 实际上 shape=1 时 stride 应该被忽略，或者说是广播。
                    // 这种情况下 merge 通常是安全的，只要我们更新 size
                    // 但标准的连续性检查公式 prevStride == currStride * 1 仍然适用。
                    // 如果广播（stride=0），则 0 == 0 * 1 成立。
                }

                if (prevStride !== currStride * currDimSize) {
                    canMerge = false;
                    break;
                }
            }

            if (canMerge) {
                // 合并维度
                currDimSize *= prevDimSize;
                // Strides 保持为内层 (current) 的 strides
            } else {
                // 无法合并，保存当前的块
                newShape.unshift(currDimSize);
                for (let k = 0; k < operands.length; k++) {
                    newStridesList[k].unshift(currStrides[k]);
                }

                // 重置为上一层
                currDimSize = prevDimSize;
                currStrides = prevStrides;
            }
        }

        // 保存最后一个块
        newShape.unshift(currDimSize);
        for (let k = 0; k < operands.length; k++) {
            newStridesList[k].unshift(currStrides[k]);
        }

        // 如果维度没有减少，说明没有优化空间 (或者没必要改变)
        if (newShape.length === ndim) {
            return undefined;
        }

        // 应用修改
        for (let k = 0; k < operands.length; k++) {
            // 我们需要 cast 因为 strides 在接口定义中经常是 readonly
            (operands[k].strides as any) = newStridesList[k];
        }

        return { shape: newShape };
    }
}
