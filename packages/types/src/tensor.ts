import { DeviceNameEnum, DType, MemoryFormat, Shape, TensorData, TypePromotionKind } from "./base";
import { IStorage, StorageBufferType } from "./storage";

export const TensorHandleSymbol = Symbol.for("kandle.tensor.handle")

export interface ITensorHandle {

    readonly shape: Shape;

    readonly dtype: DType;

    readonly buffer: StorageBufferType;

    readonly numel: number;

    readonly strides: readonly number[];

    readonly offset: number;

    readonly id: number;

    readonly storage: IStorage;

    readonly storageId: number;

    readonly device: DeviceNameEnum;

    /**
     * 内存布局格式
     *
     * 决定多维数据在线性内存中的排列顺序。
     * 这是逻辑概念，通过 strides 物理实现。
     *
     * - Contiguous: 标准 row-major (NCHW for 4D)
     * - ChannelsLast: 通道维度最密集 (NHWC physical layout, 4D only)
     * - ChannelsLast3d: 5D 版本 (NDHWC)
     *
     * @default MemoryFormat.Contiguous
     */
    readonly memoryFormat: MemoryFormat;

    readonly [TensorHandleSymbol]: true;

    /**
     * Asynchronously retrieve tensor data with backend-specific type conversion.
     * For example, WebGPU backend may need to convert from physical storage format
     * (e.g., f32 for float64 tensors) back to the logical type.
     */
    dataAsync(): Promise<TensorData>;

    dispose(): void;

    isDisposed: boolean;

}

export interface TensorOptions {
    /**
     * The data type of the tensor. Defaults to 'float32'.
     */
    dtype?: DType;
    /**
     * The device to store the tensor on. Defaults to the current active backend device.
     */
    device?: DeviceNameEnum;
    /**
     * Explicitly specify the shape of the tensor.
     * Useful when creating from a flat TypedArray or when you want to reshape immediately.
     * If not provided, the shape is inferred from the data.
     */
    shape?: Shape;
}

// ============================================================================
// TensorIterator V2 - New Architecture Types
// ============================================================================

/**
 * TensorIterator 构建配置
 *
 * 这是构建 TensorIterator 的唯一输入，所有信息一次性传入。
 * 在 build() 时统一校验和处理。
 *
 * 设计目标：
 * - 配置对象化：用单一对象传入所有参数，避免链式调用的状态管理问题
 * - 语义清晰：每个字段的含义明确，无二义性
 * - 灵活性：通过可选字段支持从 unary 到 reduction 的各种操作
 */
export interface TensorIteratorConfig {
    // ===========================================
    // 操作数（内部会按 outputs-first 排列）
    // ===========================================

    /**
     * 输入张量列表
     *
     * 长度决定操作类型：
     * - 0 个: nullary (如 zeros, ones, full)
     * - 1 个: unary (如 abs, neg, sqrt) 或 reduction (如 sum, mean)
     * - 2 个: binary (如 add, sub, mul)
     * - 3 个: ternary (如 where, addcmul, lerp)
     */
    inputs: ITensorHandle[];

    /**
     * 输出张量列表
     *
     * - undefined 表示需要自动分配新张量
     * - 已有张量表示 in-place 或 out= 参数
     * - 多输出时（如 argmax 返回 [values, indices]）数组长度 > 1
     *
     * 示例:
     * - [undefined]: 自动分配单输出
     * - [out]: 使用用户提供的输出张量
     * - [undefined, undefined]: 双输出（如 max(dim) 返回值和索引）
     */
    outputs: (ITensorHandle | undefined)[];

    // ===========================================
    // 操作信息
    // ===========================================

    /**
     * 操作名称
     *
     * 用于日志输出和 kernel 命名。
     * 类型推导现在优先使用 typePromotionKind 参数。
     */
    opName: string;

    /**
     * 类型推导策略
     *
     * 直接传入类型推导规则，避免通过 opName 反查 OpRegistry。
     * 这遵循 PyTorch TensorIteratorConfig 的设计模式。
     *
     * 如果不提供，将回退到通过 opName 查询 OpRegistry（可能失败）。
     *
     * @see TypePromotionKind
     */
    typePromotionKind?: TypePromotionKind;

    // ===========================================
    // Reduction 配置（可选）
    // ===========================================

    /**
     * Reduction 配置
     *
     * 如果提供，则表示这是一个 reduction 操作（sum, mean, max 等）。
     * 不提供则表示 elementwise 操作。
     */
    reduction?: {
        /**
         * 要归约的轴
         *
         * 数组形式，支持多轴同时归约。
         * 已规范化为正数索引（0-based）。
         */
        axes: number[];

        /**
         * 是否保持归约维度为 1
         *
         * true: 输出形状与输入形状维度数相同，归约维度为 1
         * false: 输出形状移除归约维度
         *
         * 示例: input.shape = [2, 3, 4], axes = [1]
         * - keepDims=true: output.shape = [2, 1, 4]
         * - keepDims=false: output.shape = [2, 4]
         */
        keepDims: boolean;
    };

    // ===========================================
    // 静态声明（用于 nullary 或强制覆盖）
    // ===========================================

    /**
     * 静态形状声明
     *
     * 使用场景：
     * - nullary 操作（无输入时必须指定，如 zeros([2,3])）
     * - 绕过广播计算（用户保证形状正确时可强制指定）
     */
    staticShape?: Shape;

    /**
     * 静态 dtype 声明
     *
     * 使用场景：
     * - nullary 操作（如 zeros(shape, dtype)）
     * - 强制指定 common dtype（绕过自动类型提升）
     */
    staticDtype?: DType;

    /**
     * 每个输出的 dtype 覆盖
     *
     * 数组索引对应 outputs 数组。
     * undefined 表示从 opName 的 resultType 规则推断。
     *
     * 典型使用场景：
     * - argmax/argmin: [undefined, 'int64'] 第二输出固定为 int64 索引
     * - max(dim): [undefined, 'int64'] 返回值保持输入类型，索引为 int64
     */
    outputDtypes?: (DType | undefined)[];

    // ===========================================
    // 行为开关（默认值符合大多数场景）
    // ===========================================

    /**
     * 检查所有张量在同一设备
     *
     * 默认 true。跨设备操作需要显式数据迁移，不支持隐式转换。
     */
    checkAllSameDevice?: boolean;

    /**
     * 检查所有输入 dtype 相同
     *
     * 默认 true。类型不同时会抛出错误，除非启用 promoteInputsToCommonDtype。
     * 注意：这是检查，不是强制转换。
     */
    checkAllSameDtype?: boolean;

    /**
     * 检查内存重叠
     *
     * 默认 true。检测输入输出之间的非安全内存重叠。
     * 安全重叠（完全相同的 view）允许，部分重叠禁止。
     */
    checkMemOverlap?: boolean;

    /**
     * 将输入提升到公共 dtype
     *
     * 默认 false。启用后，不同类型的输入会自动提升到能容纳所有输入的最小公共类型。
     * 遵循 PyTorch 类型提升规则：Bool < Int < Float < Complex
     */
    promoteInputsToCommonDtype?: boolean;

    /**
     * 整数输入提升到浮点
     *
     * 默认 false。用于除法、sqrt、log 等必须在浮点域计算的操作。
     * 启用后，整数输入会被提升到默认浮点类型（float32）。
     */
    promoteIntegerInputsToFloat?: boolean;

    /**
     * 强制安全类型转换到输出
     *
     * 默认 false。启用后，如果结果类型无法无损转换到输出类型，会抛出错误。
     * 例如：float32 结果写入 int32 输出会报错。
     */
    enforceSafeCastingToOutput?: boolean;

    /**
     * 禁用维度折叠优化
     *
     * 默认 false。启用后，跳过 coalesceDimensions 优化。
     *
     * 使用场景：
     * - copy 操作时输入输出内存布局不同（如 NCHW -> NHWC 转换）
     * - 维度折叠会错误地合并具有不同 strides 模式的维度
     */
    disableDimensionCoalescing?: boolean;
}

/**
 * 单个操作数的完整信息
 *
 * 设计目标：
 * - 包含 tensorHandle 引用，使结果可以通过 iter.output().tensorHandle 获取
 * - 提供 kernel 执行所需的所有信息（buffer, strides, offset 等）
 * - 支持 reduction 的双步长设计（parallel strides + reduction strides）
 *
 * 替代原 IOperandConfigBase，语义更完整。
 */
export interface TensorIteratorOperand {
    /**
     * 操作数名称
     *
     * 用于调试日志和 shader codegen 中的变量命名。
     * 约定：输入使用 'input0', 'input1'...; 输出使用 'output0', 'output1'...
     */
    name: string;

    /**
     * 对应的 TensorHandle
     *
     * **关键设计**：对于自动分配的输出，这里是新创建的 handle。
     * 用户通过 iter.output().tensorHandle 获取计算结果，
     * 无需在 dispatch 层额外维护 resultHandle 变量。
     */
    tensorHandle: ITensorHandle;

    /**
     * 底层 buffer
     *
     * 直接来自 tensorHandle.storage.buffer。
     * 冗余存储以避免 kernel 中重复访问。
     * 类型为 GPUBuffer（WebGPU）或 TypedArray（JS）。
     */
    buffer: StorageBufferType;

    /**
     * 数据类型
     *
     * 操作数的逻辑 dtype。可能与 commonDtype 不同，
     * kernel 负责在必要时进行类型转换。
     */
    dtype: DType;

    /**
     * 原始形状
     *
     * 操作数的原始形状（广播前）。
     * 用于日志和某些需要原始形状信息的 kernel。
     */
    shape: readonly number[];

    /**
     * 广播/对齐后的 strides
     *
     * **核心字段**：这是 kernel 遍历数据的关键。
     *
     * 对于输入：
     * - 经过广播调整，维度扩展处 stride=0
     * - 已对齐到 outputShape 的维度数
     *
     * 对于输出：
     * - 正常 row-major strides（或用户指定的 strides）
     *
     * 单位：元素数（不是字节数）
     */
    strides: readonly number[];

    /**
     * 在 storage 中的字节偏移量
     *
     * 用于 view 操作产生的非零起始位置。
     * kernel 访问第 i 个元素时需要加上此偏移。
     *
     * 单位：字节
     */
    offset: number;

    /**
     * 是否是输出操作数
     *
     * true: 此操作数是 kernel 写入目标
     * false: 此操作数是只读输入
     *
     * 用于 binding type 决策和内存重叠检测。
     */
    isOutput: boolean;

    // ===========================================
    // Reduction 扩展（仅输入操作数）
    // ===========================================

    /**
     * Reduction 维度的 strides
     *
     * 仅在 isReduction=true 且 isOutput=false 时有值。
     * 表示在归约维度上遍历元素的步长。
     *
     * 与 strides 配合实现双循环：
     * - 外循环使用 strides 遍历输出位置
     * - 内循环使用 reductionStrides 遍历要归约的元素
     *
     * 单位：元素数
     */
    reductionStrides?: readonly number[];
}

/**
 * TensorIterator - 张量操作的迭代抽象
 *
 * **设计理念**：
 * - 一旦构建完成，此对象不可变（immutable）
 * - Kernel 从此对象读取所有需要的信息来执行计算
 * - outputs-first 排列：operands 数组中输出在前，输入在后
 *
 * **使用模式**：
 * ```typescript
 * const iter = TensorIterator.binaryOp(a, b, 'add', out);
 * backend.operators.find('add')(iter);
 * return iter.output().tensorHandle;
 * ```
 */
export interface ITensorIterator {
    // ===========================================
    // 形状信息（语义清晰分离）
    // ===========================================

    /**
     * 输入形状（广播后的公共形状）
     *
     * - Elementwise: 所有输入广播后的形状，等于输出形状
     * - Reduction: 广播后、收缩前的形状（原始输入维度）
     *
     * 用于 reduction 内循环确定归约范围。
     */
    readonly inputShape: readonly number[];

    /**
     * 输出形状
     *
     * - Elementwise: 等于 inputShape
     * - Reduction: 收缩后形状（受 keepDims 影响）
     *
     * 这是 kernel 外循环的迭代范围，也是输出张量的实际形状。
     */
    readonly outputShape: readonly number[];

    /**
     * 输出元素数
     *
     * outputShape 的乘积。这是 kernel dispatch 的并行单元数。
     */
    readonly outputNumel: number;

    // ===========================================
    // 类型信息
    // ===========================================

    /**
     * 公共 dtype
     *
     * 输入类型提升后的结果。用于确定 kernel 的主要数据类型。
     * 例如：int32 + float32 → float32
     */
    readonly commonDtype: DType;

    /**
     * 计算 dtype
     *
     * 中间计算使用的类型，通常等于 commonDtype。
     * 特殊情况：
     * - sum 整数时可能用更大的类型累加（避免溢出）
     * - mean 整数时需要用浮点累加
     */
    readonly computeDtype: DType;

    // ===========================================
    // 操作数（outputs first）
    // ===========================================

    /**
     * 所有操作数（outputs 在前，inputs 在后）
     *
     * outputs-first 排列的理由：
     * - 与 PyTorch ATen 保持一致
     * - 输出通常只有 1 个，放在固定位置便于访问
     * - 多输出时保持有序
     */
    readonly operands: readonly TensorIteratorOperand[];

    /** 输出数量 */
    readonly numOutputs: number;

    /** 输入数量 */
    readonly numInputs: number;

    /**
     * 快捷访问第 idx 个输出（默认 0）
     *
     * 等价于 operands[idx]，但更语义化。
     *
     * @example
     * const result = iter.output().tensorHandle; // 获取第一个输出
     * const indices = iter.output(1).tensorHandle; // argmax 的索引输出
     */
    output(idx?: number): TensorIteratorOperand;

    /**
     * 快捷访问第 idx 个输入（默认 0）
     *
     * 等价于 operands[numOutputs + idx]，但更语义化。
     *
     * @example
     * const inputA = iter.input(0).buffer;
     * const inputB = iter.input(1).buffer;
     */
    input(idx?: number): TensorIteratorOperand;

    // ===========================================
    // Reduction 信息
    // ===========================================

    /**
     * 是否是 reduction 操作
     *
     * true: sum, mean, max, min, prod 等归约操作
     * false: elementwise 操作（unary, binary, ternary）
     */
    readonly isReduction: boolean;

    /**
     * Reduction 的轴索引（已规范化为正数）
     *
     * 例如：input.shape = [2, 3, 4], axes = [1, 2]
     *       reductionAxes = [1, 2]
     *
     * 用于 keepDims=true 时确定哪些输出维度对应 reduced dimensions。
     * 非 reduction 操作时为空数组 []。
     */
    readonly reductionAxes: readonly number[];

    /**
     * 是否保持 reduction 维度（大小为 1）
     *
     * true: 输出形状保持与输入相同的维度数，reduced 维度大小为 1
     * false: 移除 reduced 维度
     *
     * 非 reduction 操作时为 false。
     */
    readonly keepDims: boolean;

    /**
     * Reduction 维度形状
     *
     * 每个输出元素要归约的维度大小。
     * 例如：input.shape = [2, 3, 4], axes = [1, 2]
     *       reductionShape = [3, 4]
     *
     * 非 reduction 操作时为空数组 []。
     */
    readonly reductionShape: readonly number[];

    /**
     * Reduction 元素数
     *
     * reductionShape 的乘积。每个输出元素需要归约的输入元素数。
     *
     * 非 reduction 操作时为 1。
     */
    readonly reductionNumel: number;

    // ===========================================
    // 优化标记
    // ===========================================

    /**
     * 所有操作数是否内存连续
     *
     * true: 所有操作数都是 row-major 连续存储，offset=0
     * false: 存在非连续视图或非零 offset
     *
     * 连续时可使用更高效的 flat loop kernel。
     */
    readonly isContiguous: boolean;

    /**
     * 是否可以安全执行 in-place
     *
     * True 当且仅当：所有与输出共享 buffer 的输入
     * 都具有完全相同的 geometry (strides + offset)。
     *
     * **WebGPU 限制**：即使 isSafeInplace=true，
     * 由于 WebGPU 规范限制，仍需创建临时 buffer。
     * 此标记用于未来优化或其他后端。
     */
    readonly isSafeInplace: boolean;
}
