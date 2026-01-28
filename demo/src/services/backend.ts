import { env } from '@kandle/core';
import { WebGPUBackend } from '@kandle/backend-webgpu';

export interface WebGPUInfo {
  available: boolean;
  supportsF16: boolean;
  limits: {
    maxBufferSize: number;
    maxStorageBufferBindingSize: number;
    maxComputeWorkgroupStorageSize: number;
    maxComputeInvocationsPerWorkgroup: number;
  };
}

let backendInitialized = false;
let webgpuInfo: WebGPUInfo | null = null;

/**
 * Initialize WebGPU backend
 *
 * WebGPUBackend.create() internally calls WebGPUDeviceManager.init()
 * which handles all WebGPU initialization
 */
export async function initializeBackend(): Promise<WebGPUInfo> {
  if (backendInitialized && webgpuInfo) {
    return webgpuInfo;
  }

  // Check WebGPU support first
  if (!('gpu' in navigator)) {
    webgpuInfo = {
      available: false,
      supportsF16: false,
      limits: {
        maxBufferSize: 0,
        maxStorageBufferBindingSize: 0,
        maxComputeWorkgroupStorageSize: 0,
        maxComputeInvocationsPerWorkgroup: 0,
      },
    };
    throw new Error('WebGPU is not supported in this browser');
  }

  try {
    // WebGPUBackend.create() handles all initialization internally
    const backend = await WebGPUBackend.create();
    env.setBackend(backend);
    env.setDefaultDevice(backend.name);

    // After backend creation, WebGPUDeviceManager is initialized
    // We can get device info from console logs or set basic info
    webgpuInfo = {
      available: true,
      supportsF16: false, // Will be logged by WebGPUDeviceManager
      limits: {
        maxBufferSize: 0,
        maxStorageBufferBindingSize: 0,
        maxComputeWorkgroupStorageSize: 0,
        maxComputeInvocationsPerWorkgroup: 0,
      },
    };

    backendInitialized = true;
    console.log('[Kandle] Backend initialized:', backend.name);

    return webgpuInfo;
  } catch (error) {
    webgpuInfo = {
      available: false,
      supportsF16: false,
      limits: {
        maxBufferSize: 0,
        maxStorageBufferBindingSize: 0,
        maxComputeWorkgroupStorageSize: 0,
        maxComputeInvocationsPerWorkgroup: 0,
      },
    };
    throw error;
  }
}

/**
 * Get current WebGPU info
 */
export function getWebGPUInfo(): WebGPUInfo | null {
  return webgpuInfo;
}

/**
 * Check if backend is initialized
 */
export function isBackendInitialized(): boolean {
  return backendInitialized;
}
