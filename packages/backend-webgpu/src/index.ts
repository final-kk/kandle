/**
 * WebGPU Backend
 * 
 * v6 架构: 所有 kernel 通过 operators 注册表访问
 * 移除了 executeEye/executeTriangular 等直接暴露的执行函数
 */

import { IBackend, ITensorHandle, DType, DeviceNameEnum, Shape, TensorData, IBackendOpsRegister, CreateTensorOptions } from "@kandle/types";
import { WebGPUTensor } from "./base/tensor";
import { WebGPUDeviceManager } from "./base/device";
import { SimpleOperatorRegistry, computeStrides } from "@kandle/utils";
import { registerWebGPUKernels } from "./kernels";
import { WebGPUStorage } from "./base/storage";

export class WebGPUBackend implements IBackend {

    name: DeviceNameEnum = DeviceNameEnum.WebGPU;

    operators: IBackendOpsRegister = new SimpleOperatorRegistry();

    constructor() {
        registerWebGPUKernels(this.operators);
    }

    /**
     * 创建 TensorHandle（统一入口）
     * 
     * 支持两种调用方式:
     * 1. createTensorHandle(shape, dtype, data?)  - 分配新 storage
     * 2. createTensorHandle(options)              - 完整选项，支持 view 和 memoryFormat
     */
    createTensorHandle(
        shapeOrOptions: Shape | CreateTensorOptions,
        dtype?: DType,
        data?: TensorData
    ): ITensorHandle {
        // 检测调用方式
        if (Array.isArray(shapeOrOptions)) {
            // 旧式调用: createTensorHandle(shape, dtype, data?)
            return WebGPUTensor.createNew(shapeOrOptions, dtype!, data as any);
        }

        // 新式调用: createTensorHandle(options)
        const options = shapeOrOptions as CreateTensorOptions;

        if (options.storage) {
            // View 模式: 共享现有 storage
            const strides = options.strides ?? computeStrides(options.shape);
            const offset = options.offset ?? 0;
            return WebGPUTensor.createView(
                options.storage as WebGPUStorage,
                options.shape,
                options.dtype,
                strides,
                offset,
                options.memoryFormat  // 传递 memoryFormat
            );
        } else {
            // 分配新 storage
            return WebGPUTensor.createNew(
                options.shape,
                options.dtype,
                options.data as any,
                options.memoryFormat  // 传递 memoryFormat
            );
        }
    }

    static _webgpuDevice: GPUDevice | null = null;

    static _instance: WebGPUBackend | null = null;

    static async create(): Promise<IBackend> {

        if (this._instance) {
            console.warn('WebGPUBackend is already created.');
            return this._instance;
        }

        await WebGPUDeviceManager.init();

        this._instance = new WebGPUBackend();

        return this._instance;
    }

}

// Export WindowFunc types for use in dispatch handlers
export type {
    WindowFuncKernelArgs,
    WindowConfig,
    GeneralizedCosineConfig,
    LinearConfig,
    KaiserConfig,
    GaussianConfig,
    ExponentialConfig,
} from './kernels/windowfunc';

// Export IIR types for use in dispatch handlers
export type { BiquadCoeffs, IIRScanParams, IIRBiquadKernelArgs } from './kernels/audio';
