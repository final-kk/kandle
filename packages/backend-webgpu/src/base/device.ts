/**
 * WebGPU Device Manager
 * 
 * 管理 WebGPU 设备的初始化和功能检测
 */

import { UniformBufferPool } from "./UniformBufferPool";
import { initGlobalDTypeResolver, resetGlobalDTypeResolver } from "./DTypeResolver";

export class WebGPUDeviceManager {

    private static _device: GPUDevice | null = null;
    private static _adapter: GPUAdapter | null = null;

    // 功能标志
    private static _supportsF16: boolean = false;

    // GPU Limits (缓存以避免重复查询)
    private static _limits: GPUSupportedLimits | null = null;

    // 预计算的最优配置（在init时计算，榨干显卡性能）
    private static _optimalWorkgroupSize: number = 0;
    private static _maxSharedMemoryBytes: number = 0;

    static get device(): GPUDevice {
        if (!this._device) {
            throw new Error("WebGPU device has not been initialized. Call init() first.");
        }
        return this._device;
    }

    static get adapter(): GPUAdapter {
        if (!this._adapter) {
            throw new Error("WebGPU adapter has not been initialized. Call init() first.");
        }
        return this._adapter;
    }

    /**
     * 获取GPU limits（用于动态调整workgroup size等）
     */
    static get limits(): GPUSupportedLimits {
        if (!this._limits) {
            throw new Error("WebGPU device has not been initialized. Call init() first.");
        }
        return this._limits;
    }

    /**
     * 检查当前设备是否支持 shader-f16 扩展
     * 
     * 如果支持，在 Shader 中可以使用 f16 类型，这对于某些 ML 模型可以提高性能
     */
    static get supportsF16(): boolean {
        return this._supportsF16;
    }

    /**
     * 获取最优workgroup size（预计算，榨干显卡性能）
     * 返回最大的2的幂 <= maxComputeInvocationsPerWorkgroup
     * 
     * 标准保证：最少256（兼容模式128）
     * 现代GPU可能支持：512, 1024, 2048+
     */
    static get optimalWorkgroupSize(): number {
        if (this._optimalWorkgroupSize === 0) {
            throw new Error("WebGPU device has not been initialized. Call init() first.");
        }
        return this._optimalWorkgroupSize;
    }

    /**
     * 获取workgroup可用的最大shared memory大小（字节）
     * 用于tree reduction的shared memory分配
     */
    static get maxSharedMemoryBytes(): number {
        if (!this._limits) {
            throw new Error("WebGPU device has not been initialized. Call init() first.");
        }
        return this._maxSharedMemoryBytes;
    }

    /**
     * 计算最大的2的幂 <= n
     * 用于榨干显卡性能（使用最大可用的2次幂workgroup size）
     */
    private static largestPowerOf2(n: number): number {
        let power = 1;
        while (power * 2 <= n) {
            power *= 2;
        }
        return power;
    }

    static async init(): Promise<void> {
        if (this._device) {
            return;
        }

        if (!navigator.gpu) {
            throw new Error("WebGPU is not supported in this environment.");
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            throw new Error("Failed to get GPU adapter.");
        }
        this._adapter = adapter;

        // 检测 shader-f16 支持
        const features = adapter.features;
        this._supportsF16 = features.has('shader-f16');

        console.log(`[WebGPU] Adapter features:`, [...features]);
        console.log(`[WebGPU] shader-f16 support: ${this._supportsF16}`);

        // 请求设备时，如果支持 f16，则请求该功能
        const requiredFeatures: GPUFeatureName[] = [];
        if (this._supportsF16) {
            requiredFeatures.push('shader-f16');
        }

        // 获取适配器支持的最大缓冲区大小
        const adapterLimits = adapter.limits;
        const maxBufferSize = adapterLimits.maxBufferSize;

        console.log(`[WebGPU] Adapter maxBufferSize: ${maxBufferSize} (${(maxBufferSize / 1024 / 1024).toFixed(0)}MB)`);

        this._device = await adapter.requestDevice({
            requiredFeatures,
            requiredLimits: {
                // 请求适配器支持的最大缓冲区大小（用于大模型权重）
                maxBufferSize: maxBufferSize,
                // 请求更大的存储缓冲区绑定大小
                maxStorageBufferBindingSize: Math.min(maxBufferSize, adapterLimits.maxStorageBufferBindingSize),
            },
        });

        // 缓存limits
        this._limits = this._device.limits;

        // 预计算最优workgroup size（榨干显卡）
        const maxInvocations = this._limits.maxComputeInvocationsPerWorkgroup;
        this._optimalWorkgroupSize = this.largestPowerOf2(maxInvocations);

        // 缓存shared memory限制
        this._maxSharedMemoryBytes = this._limits.maxComputeWorkgroupStorageSize;

        console.log(`[WebGPU] Device limits:`, {
            maxBufferSize: this._limits.maxBufferSize,
            maxStorageBufferBindingSize: this._limits.maxStorageBufferBindingSize,
            maxComputeWorkgroupSizeX: this._limits.maxComputeWorkgroupSizeX,
            maxComputeWorkgroupSizeY: this._limits.maxComputeWorkgroupSizeY,
            maxComputeWorkgroupSizeZ: this._limits.maxComputeWorkgroupSizeZ,
            maxComputeInvocationsPerWorkgroup: this._limits.maxComputeInvocationsPerWorkgroup,
            maxComputeWorkgroupsPerDimension: this._limits.maxComputeWorkgroupsPerDimension,
            maxComputeWorkgroupStorageSize: this._limits.maxComputeWorkgroupStorageSize,
        });

        // console.log(`[WebGPU] Computed optimal workgroup size: ${this._optimalWorkgroupSize} (榨干显卡: max available ${maxInvocations})`);

        this._device.lost.then((info) => {
            console.error("WebGPU device lost:", info);
            this._device = null;
        });



        // 初始化 UniformBufferPool
        UniformBufferPool.initialize(this._device);

        // 初始化 DType 解析器 (根据设备能力确定物理存储策略)
        initGlobalDTypeResolver(this._supportsF16);

        console.log(`[WebGPU] Device initialized with features:`, requiredFeatures);
    }

    /**
     * 获取 float16 的实际存储类型
     * 
     * 如果设备支持 f16，返回 'f16'；否则返回 'f32' 作为回退
     */
    static getFloat16StorageType(): 'f16' | 'f32' {
        return this._supportsF16 ? 'f16' : 'f32';
    }

    /**
     * 重置设备状态（用于测试）
     */
    static reset(): void {


        // 清理 UniformBufferPool
        UniformBufferPool.reset();

        // 重置 DType 解析器
        resetGlobalDTypeResolver();

        if (this._device) {
            this._device.destroy();
            this._device = null;
        }
        this._adapter = null;
        this._supportsF16 = false;
        this._limits = null;
        this._optimalWorkgroupSize = 0;
        this._maxSharedMemoryBytes = 0;
    }
}
