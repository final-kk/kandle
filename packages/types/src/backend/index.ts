import { DeviceNameEnum, DType, MemoryFormat, Shape, TensorData } from "../base";
import { ITensorHandle } from "../tensor";
import { IBackendOpsRegister } from "./register";
import { IStorage } from "../storage";

/**
 * TensorHandle 创建选项
 */
export interface CreateTensorOptions {
    /** 目标形状 */
    shape: Shape;
    /** 数据类型 */
    dtype: DType;
    /** 
     * 用于 View 的共享 storage (可选)
     * 如果提供，则创建共享 storage 的 view tensor，否则分配新 storage
     */
    storage?: IStorage;
    /** 初始数据 (仅在不使用 storage 时有效) */
    data?: TensorData;
    /** 自定义 strides (仅在使用 storage 时有效) */
    strides?: number[];
    /** 在 storage 中的偏移量 (仅在使用 storage 时有效) */
    offset?: number;
    /** 
     * 内存布局格式 (可选)
     * 如果提供，将影响 strides 的计算方式
     * @default MemoryFormat.Contiguous
     */
    memoryFormat?: MemoryFormat;
}

export interface IBackend {
    name: DeviceNameEnum;

    /**
     * 创建 TensorHandle（统一入口）
     * 
     * 支持两种模式:
     * 1. 分配新 storage: 仅提供 shape, dtype, data
     * 2. 共享现有 storage (View): 提供 storage, shape, dtype, strides, offset
     */
    createTensorHandle(options: CreateTensorOptions): ITensorHandle;

    /**
     * @deprecated 使用 createTensorHandle({ shape, dtype, data }) 替代
     * 为向后兼容保留的重载
     */
    createTensorHandle(shape: Shape, dtype: DType, data?: TensorData): ITensorHandle;

    operators: IBackendOpsRegister;

    create?(): IBackend;
    create?(): Promise<IBackend>;
}

export * from './register';
export * from './dispatch';
