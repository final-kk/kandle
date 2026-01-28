import { create, globals } from 'webgpu';
import { env } from "@kandle/core";
import { WebGPUBackend } from "@kandle/backend-webgpu"
import { DeviceNameEnum } from '@kandle/types';

export function initWgpu() {
    Object.assign(globalThis, globals);
    const gpu = create([]);
    if (typeof (globalThis as any).navigator === 'undefined') {
        (globalThis as any).navigator = {};
    }
    Object.defineProperty((globalThis as any).navigator, 'gpu', {
        value: gpu,
        writable: true,
        enumerable: true,
        configurable: true,
    });
}

export async function initWgpuBackend() {
    env.setBackend(await WebGPUBackend.create());
    env.setDefaultDevice(DeviceNameEnum.WebGPU);
}

export async function init() {
    initWgpu();
    await initWgpuBackend();
}